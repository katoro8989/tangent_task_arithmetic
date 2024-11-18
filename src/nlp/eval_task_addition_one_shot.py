import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import torch


import argparse
from eval import eval_single_dataset, evaluate_task_vector, evaluate_task_vector_at_coef
from task_vectors import T5LinearizedTaskVector, T5NonLinearTaskVector
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from torch.utils.data import DataLoader

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils import find_optimal_coef


parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="google/flan-t5-small")
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

args.data_dir = "/mnt2/dataset/glue_split"

args.n_eval_points = 21
args.eval_max_points = 1.0

args.tokenizer_kwargs = {
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
    }

if args.finetuning_mode == "ours":
    if args.seed is not None:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}_ours/{args.model}"
    else:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.model}_ours"
else:
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
elif args.finetuning_mode == "ours":
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies_ours.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)

# with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
#     pretrained_accuracies = json.load(f)

tokenizer = T5Tokenizer.from_pretrained(args.model)

eval_datasets = [
    "cola",
    "mrpc",
    "rte",
    "sst2",
]

task_vectors = []

for dataset in eval_datasets:
    if args.finetuning_mode == "linear" or args.finetuning_mode == "ours":
        pretrained_checkpoint = f"{args.save}/{dataset}/linear_zeroshot"
        finetuned_checkpoint = f"{args.save}/{dataset}/linear_finetuned"
        task_vectors.append(
            T5LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot"
        finetuned_checkpoint = f"{args.save}/{dataset}/finetuned"
        task_vectors.append(
            T5NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )

task_vector = sum(task_vectors)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    tokenizer, 
    args,
    1.0,
    posthoc_linearization=args.finetuning_mode == "posthoc",
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions_one_shot.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions_one_shot.json"
elif args.finetuning_mode == "ours":
    save_file = f"{args.save}/ours_additions_one_shot.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)


