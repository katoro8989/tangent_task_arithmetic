import os
import sys
import time
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
from dataset_preprocess.glue_process import *
import argparse
import datetime
import wandb
import evaluate
import uuid
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.utils import LabelSmoothing, cosine_lr




from linearize import LinearizedModel, LinearizedModelWraper, SimpleCallableT5Model

preprocessor_mapping = {
    "cola": CoLA_Preprocessor,
    "rte": RTE_Preprocessor,
    "mnli": MNLI_Preprocessor,
    "mrpc": MRPC_Preprocessor,
    "qnli": QNLI_Preprocessor,
    "qqp": QQP_Preprocessor,
    "sst2": SST2_Preprocessor,
    "stsb": STSB_Preprocessor,
}

tokenizer_kwargs = {
  "padding": "max_length",
  "truncation": True,
  "return_tensors": "pt",
}

map_kwargs = {
    "remove_columns": ["sentence", "label", "idx"],
    "batched": True,
    "num_proc": 1,
    "desc": "Running tokenizer on dataset"
}

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
    
    run_id = f'{args.model}'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f'{run_id}_{timestamp}'
    output_dir = os.path.join(args.output_dir, run_id)

    
    hf_t5_model = T5ForConditionalGeneration.from_pretrained(args.model)
    model = simple_model_class = SimpleCallableT5Model(hf_t5_model)

    if args.ft_method == "linear":
        linearized_finetuning = True
        model = LinearizedModelWraper(model)
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    dataset_class = load_dataset("glue", args.task)
    
    preprocessor_class = preprocessor_mapping[args.task]
    preprocessor = preprocessor_class(tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs)
    encoded_dataset = dataset_class.map(preprocessor, **map_kwargs)

    # DataLoaderの作成
    train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size)

    
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

    loss_fn = torch.nn.CrossEntropyLoss()

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
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ddp_model.module.model.save(model_path)

    print_every = 100

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

            loss = loss_fn(logits, labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                ddp_model.module.model.save(model_path)

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_train_loader)

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_train_loader)}]\t"  # noqa: E501
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
                    flush=True,
                )
                run.log({
                    'step': step,
                    'train_loss': loss.item(),
                })

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, "linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, "linear_finetuned.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, "finetuned.pt")
        )
        model.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning of T5')
    parser.add_argument('--task', type=str, default="cola")
    parser.add_argument('--model', type=str, default="google/flan-t5-small")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--num_grad_accumulation', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=int, default=0.01)
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
    args = parser.parse_args()

    # HACK: Some command line arguments are overwritten by defaults here.
    args.world_size = 4
    args.port = 12345
    args.seed = 42

    if args.seed is not None:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.model}"

    print("=" * 100)
    print(f"Finetuning {args.model} on {args.task}")
    print("=" * 100)

    group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))

    torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)
                