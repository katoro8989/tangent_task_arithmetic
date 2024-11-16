import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np
import argparse
import os
from tqdm import tqdm
from task_vectors import GPT2NonLinearTaskVector, GPT2LinearizedTaskVector
from torch.utils.data import DataLoader
import math

parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="gpt2")
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

if args.finetuning_mode == "ours":
    if args.seed is not None:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.seed}_ours/{args.model}"
    else:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.model}_ours"
else:
    if args.seed is not None:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.seed}/{args.model}"
    else:
        args.save = f"/mnt2/gpt2_civil_checkpoints_{args.model}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

pretrained_checkpoint = (
    f"{args.save}/linear_zeroshot"
    if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
    else f"{args.save}/zeroshot"
)

finetuned_checkpoint = (
    f"{args.save}/linear_finetuned"
    if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
    else f"{args.save}/finetuned"
)

try:
    task_vector = (
        GPT2LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint, len_tokenizer=len(tokenizer))
        if args.finetuning_mode == "linear" or args.finetuning_mode == "ours"
        else GPT2NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint, len_tokenizer=len(tokenizer))
    )
except FileNotFoundError:
    print(f"Error: Could not find {finetuned_checkpoint}.")

if args.finetuning_mode == "none":
    model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear" or args.finetuning_mode == "ours":
    model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=-1.0)

model = model.to(device)
model.eval()

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

# トークナイズとデータのグループ化
block_size = 1024

def tokenize_function(examples):
    return tokenizer(examples['text'])

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=['text'],
)

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        return {'input_ids': [], 'attention_mask': []}

    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=4,
)

# 空の入力を除去
lm_datasets = lm_datasets.filter(lambda x: len(x['input_ids']) > 0)

eval_dataloader = DataLoader(lm_datasets, batch_size=args.eval_batch_size)

# パープレキシティの計算
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for batch in tqdm(eval_dataloader, desc="Calculating perplexity"):
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # 損失をトークン数で重み付け
        labels = batch['labels']
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

perplexity = math.exp(total_loss / total_tokens)
print(f"Perplexity: {perplexity}")