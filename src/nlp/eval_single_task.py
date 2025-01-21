import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_from_disk
import torch


import argparse
from eval import eval_single_dataset
from linearize import LinearizedT5Wrapper
from task_vectors import T5LinearizedTaskVector, T5NonLinearTaskVector
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="google/flan-t5-small")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="linear")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

args.tokenizer_kwargs = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
    }

if args.seed is not None:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.model}"

if args.seed is not None:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}_ours/{args.model}"
else:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.model}_ours"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")

for dataset in [
    # "cola",
    # "mrpc",
    # "rte",
    "sst2",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    args.task = dataset

    pretrained_checkpoint = (
        f"{args.save}/{dataset}/linear_zeroshot"
        if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
        else f"{args.save}/{dataset}/zeroshot"
    )

    finetuned_checkpoint = (
        f"{args.save}/{dataset}/linear_finetuned"
        if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
        else f"{args.save}/{dataset}/finetuned"
    )

    try:
        task_vector = (
            T5LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
            else T5NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    if args.finetuning_mode == "none":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    
    model.to("cuda")

    tokenizer = T5Tokenizer.from_pretrained(args.model)

    encoded_dataset = load_from_disk(f"/mnt2/dataset/glue_split/{args.task}")

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
    
    if args.task == "mnli":
        val_dataloader = DataLoader(encoded_dataset["validation_matched"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
        test_dataloader = DataLoader(encoded_dataset["test_matched"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
    else:
        val_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
        test_dataloader = DataLoader(encoded_dataset["test"], batch_size=args.eval_batch_size, collate_fn=collate_fn)

    for split in ["validation", "test"]:
        eval_dataloader = val_dataloader if split == "validation" else test_dataloader
        eval_dataset = dataset if split == "test" else f"{dataset}Val"
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        accuracies[eval_dataset] = eval_single_dataset(
            model, tokenizer, eval_dataloader, args
        )["top1"]


# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
elif args.finetuning_mode == "linear":
    # save_path = f"{args.save}/linear_ft_accuracies.json"
    save_path = f"{args.save}/linear_ft_accuracies_debug.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
