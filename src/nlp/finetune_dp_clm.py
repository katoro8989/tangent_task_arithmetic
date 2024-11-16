import os
import sys
import time
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk
import argparse
import datetime
import wandb
import uuid
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, matthews_corrcoef

from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from linearize import LinearizedGPT2Wrapper, SimpleCallableHFModel

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils import cosine_lr

def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    ckpdir = args.save

    run = wandb.init(config=vars(args),
                        project=f"{args.model}_CivilCom_{args.finetuning_mode}",
                        entity='katoro13',
                        name=f"process_{rank}",
                        group=group, 
                        )

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model = SimpleCallableHFModel(model)

    if args.finetuning_mode == "linear":
        linearized_finetuning = True
        model = LinearizedGPT2Wrapper(model)
    else:
        linearized_finetuning = False
        

    encoded_dataset = load_from_disk("/mnt2/dataset/civil_comments")

    max_step = ((len(encoded_dataset["train"]) // args.train_batch_size) * args.epochs) // 2
    args.max_steps = max_step

    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=data_collator)
    
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
            labels = input_ids.clone()
            labels[:, 0] = -100
            data_time = time.time() - start_time

            # モデルの出力を取得
            logits = ddp_model(input_ids=input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
                        labels = input_ids.clone()
                        labels[:, 0] = -100  # 最初のトークンを無視

                        data_time = time.time() - start_time

                        # モデルの出力を取得
                        logits = ddp_model(input_ids=input_ids)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                        losses.append(loss.item())


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
    parser.add_argument('--model', type=str, default="gpt2")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--num_grad_accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--warmup_length', type=int, default=0)
    parser.add_argument('--wd', type=int, default=0.1)
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

    if args.seed is not None:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.model}"
    
    print("*" * 100)
    print(f"Finetuning on Civil Comments")
    for finetuning_mode in ["linear"]:
        args.finetuning_mode = finetuning_mode
        print("*" * 100)
        print(f"Finetuning mode: {finetuning_mode}")

        group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))

        torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)
            