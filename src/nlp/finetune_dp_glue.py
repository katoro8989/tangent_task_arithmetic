import os
import sys
import time
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_from_disk
import argparse
import datetime
import wandb
import uuid
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, matthews_corrcoef

from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from linearize import LinearizedModelWrapper, SimpleCallableHFModel

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils import cosine_lr

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    train_dataset = args.task
    ckpdir = os.path.join(args.save, train_dataset)

    if "/" in args.model:
        model_name = args.model.split("/")[-1]

    run = wandb.init(config=vars(args),
                        project=f"{model_name}_GLUE_{train_dataset}_{args.finetuning_mode}",
                        entity='katoro13',
                        name=f"process_{rank}",
                        group=group, 
                        )

    
    hf_t5_model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = SimpleCallableHFModel(hf_t5_model)

    if args.finetuning_mode == "linear":
        linearized_finetuning = True
        model = LinearizedModelWrapper(model)
    else:
        linearized_finetuning = False
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    encoded_dataset = load_from_disk(f"/mnt2/dataset/glue_split/{train_dataset}")

    
    if args.task == "mnli":
        train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        eval_dataloader = DataLoader(encoded_dataset["validation_matched"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
    else:
        train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=collate_fn)

    
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

    print("Starting training.")
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

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(iter)
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
                (iter - 1) % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                ddp_model.eval()
                all_preds = []
                all_labels = []
                losses = []
                with torch.no_grad():
                    for batch in ddp_eval_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        logits = outputs

                        losses.append(loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item())


                loss_ave = sum(losses) / len(losses)
                percent_complete = (iter / max_steps) * 100

                print(
                    f"Train Step: {iter - 1} [{percent_complete:.0f}% {iter - 1}/{max_steps}]\t"  # noqa: E501
                    f"Val Loss: {loss_ave:.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )
                run.log({
                    'step': iter - 1,
                    'val_loss': loss_ave,
                    'lr': optimizer.param_groups[0]['lr'],
                })
            if  iter - 1 >= max_steps:
                if is_main_process():
                    print(f"Reached maximum steps of {max_steps}. Ending training.")
                break

        if iter - 1 >= max_steps:
            if is_main_process():
                print(f"Reached maximum steps of {max_steps}. Ending training.")
            break


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
    parser.add_argument('--wd', type=int, default=0.)
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
    parser.add_argument('--finetuning_mode', type=str, default="standard")
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
    
    for dataset in [
        "cola",
        "mrpc",
        "rte",
        "sst2",
    ]:
        print("*" * 100)
        print(f"Evaluating on {dataset}")
        for finetuning_mode in ["standard", "linear"]:
            args.finetuning_mode = finetuning_mode
            print("*" * 100)
            print(f"Finetuning mode: {finetuning_mode}")

            args.task = dataset
            print("=" * 100)
            print(f"Finetuning {args.model} on {args.task}")
            print("=" * 100)

            group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))

            torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)
                