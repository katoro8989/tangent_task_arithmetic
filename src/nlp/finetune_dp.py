import os
import sys
import time
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import argparse
import datetime
import wandb
import uuid
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, matthews_corrcoef

from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from linearize import LinearizedModelWraper, SimpleCallableT5Model

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils import cosine_lr


def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.task
    ckpdir = os.path.join(args.save, train_dataset)

    if "/" in args.model:
        model_name = args.model.split("/")[-1]

    run = wandb.init(config=vars(args),
                        project=f"{model_name}_GLUE_{train_dataset}_{args.ft_method}",
                        entity='katoro13',
                        name=f"process_{rank}",
                        group=group, 
                        )

    
    hf_t5_model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = simple_model_class = SimpleCallableT5Model(hf_t5_model)

    if args.ft_method == "linear":
        linearized_finetuning = True
        model = LinearizedModelWraper(model)
    else:
        linearized_finetuning = False
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    dataset_class = load_dataset("glue", args.task)
    
    preprocessor_class = get_preprocessor(args.task)
    preprocessor = preprocessor_class(tokenizer=tokenizer, tokenizer_kwargs=args.tokenizer_kwargs)
    map_kwargs = get_map_kwargs(args.task)
    encoded_dataset = dataset_class.map(preprocessor, **map_kwargs)

    # DataLoaderの作成
    # カスタム collate_fn の定義
    def collate_fn(batch):
        # 各バッチの要素（例: input_ids, attention_mask, labels）をテンソルに変換
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(encoded_dataset["test"], batch_size=args.eval_batch_size, collate_fn=collate_fn)

    
    # Distribute the data and model across the GPUs.
    ddp_train_loader = distribute_loader(train_dataloader)
    ddp_eval_loader = distribute_loader(eval_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    num_batches = len(encoded_dataset["train"])
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        args.epochs * num_batches // args.num_grad_accumulation,
    )

    # Saving zero-shot model
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, "linear_zeroshot")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot")
        )
        ddp_model.module.model.save_pretrained(model_path)

    print_every = 100
    max_steps = args.max_steps
    iter = 0

    for epoch in range(args.epochs):
        ddp_model.train()

        for i, batch in enumerate(ddp_train_loader):
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            data_time = time.time() - start_time

            logits = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # loss = loss_fn(logits, labels)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                iter += 1
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}")
                )
                ddp_model.module.model.save_pretrained(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                ddp_model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in ddp_eval_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs

                        preds = torch.argmax(logits, dim=-1)

                        mask = labels != -100
                        preds_valid = preds[mask]
                        labels_valid = labels[mask]

                        all_preds.extend(preds_valid.cpu().numpy())
                        all_labels.extend(labels_valid.cpu().numpy())

                # sklearn の accuracy_score を使って精度を計算
                accuracy = accuracy_score(all_labels, all_preds)
                mcc = matthews_corrcoef(all_labels, all_preds)
                percent_complete = 100 * i / len(ddp_train_loader)

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_train_loader)}]\t"  # noqa: E501
                    f"Val Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    f"Val Acc: {accuracy}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    f"Val MCC: {mcc}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )
                run.log({
                    'step': step,
                    'val_loss': loss.item(),
                    'val_accuracy': accuracy,
                    'val_mcc': mcc,
                })
            # optimizer.step() を行った後に最大ステップ数に達しているか確認
            if  iter >= max_steps:
                print(f"Reached maximum steps of {max_steps}. Ending training.")
                break  # 内側のループを終了

        # 外側のループで最大ステップ数に達しているか確認
        if iter >= max_steps:
            print(f"Reached maximum steps of {max_steps}. Ending training.")
            break  # 外側のループを終了

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot")
        )
        ft_path = (
            os.path.join(ckpdir, "linear_finetuned")
            if linearized_finetuning
            else os.path.join(ckpdir, "finetuned")
        )
        ddp_model.module.model.save_pretrained(ft_path)
        return zs_path, ft_path
    
    # evaluate on test set
    ddp_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs

            preds = torch.argmax(logits, dim=-1)

            mask = labels != -100
            preds_valid = preds[mask]
            labels_valid = labels[mask]

            all_preds.extend(preds_valid.cpu().numpy())
            all_labels.extend(labels_valid.cpu().numpy())
    

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning of T5')
    parser.add_argument('--task', type=str, default="cola")
    parser.add_argument('--model', type=str, default="google/flan-t5-small")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--num_grad_accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--warmup_length', type=int, default=0)
    parser.add_argument('--wd', type=int, default=0.01)
    parser.add_argument('--fp16', action='store_true', help='whether fp16')
    parser.add_argument('--logging_dir', type=int, default=None)
    parser.add_argument('--logging_strategy', type=str, default="steps")
    parser.add_argument('--evaluation_strategy', type=str, default="steps")
    parser.add_argument('--wandb', action='store_true', help='whether log on wandb')
    parser.add_argument('--exp_id', type=str, default=None, help='exp id for reporting')
    parser.add_argument('--fp16_full_eval', action='store_true')
    parser.add_argument('--eval_steps', type=float, default=1000)
    parser.add_argument('--eval_accumulation_steps', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--auto_find_batch_size', action='store_true')
    parser.add_argument('--ft_method', type=str, default="standard")
    parser.add_argument('--checkpoint_every', type=int, default=-1)
    args = parser.parse_args()

    # HACK: Some command line arguments are overwritten by defaults here.
    args.world_size = 4
    args.port = 12345
    args.seed = 42

    args.tokenizer_kwargs = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
    }

    if args.seed is not None:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.model}"

    print("=" * 100)
    print(f"Finetuning {args.model} on {args.task}")
    print("=" * 100)

    group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))

    torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)
                