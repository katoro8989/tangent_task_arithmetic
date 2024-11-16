import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np
import argparse
import os
from tqdm import tqdm
from task_vectors import GPT2NonLinearTaskVector, GPT2LinearizedTaskVector

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

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

def calculate_perplexity(model, tokenizer, dataset, stride=512, max_length=1024):
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Calculating perplexity"):
            input_text = sample["text"]
            encodings = tokenizer(input_text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            
            for i in range(0, input_ids.size(1), stride):
                input_slice = input_ids[:, i:i + max_length]
                if input_slice.size(1) < max_length:
                    break
                
                labels = input_slice.clone()
                outputs = model(input_slice, labels=labels)
                loss = outputs.loss
                batch_loss = loss.item() * input_slice.size(1)
                
                total_loss += batch_loss
                total_length += input_slice.size(1)
    
    avg_loss = total_loss / total_length
    perplexity = np.exp(avg_loss)
    return perplexity

perplexity = calculate_perplexity(model, tokenizer, dataset, stride=512, max_length=1024)
print(f"Perplexity: {perplexity}")