import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import torch


import argparse
from eval import eval_single_dataset
from linearize import LinearizedModelWrapper
from task_vectors import LinearizedTaskVector, NonLinearTaskVector
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils import find_optimal_coef


parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="google/flan-t5-small")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
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

accuracies = {}

TASKS = ["cola", "sst2", "mrpc", "rte"]

print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

eval_datasets = [
    "cola",
    "mrpc",
    "rte",
    "sst2",
]

task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

for dataset in [
    "cola",
    "mrpc",
    "rte",
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
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
            else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    if args.finetuning_mode == "none":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear":
        model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)

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
    save_path = f"{args.save}/linear_ft_accuracies.json"
    # save_path = f"{args.save}/linear_ft_accuracies_cars_dtd.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
