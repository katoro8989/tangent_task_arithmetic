import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import torch


import argparse
from eval import eval_single_dataset, evaluate_task_vector, evaluate_task_vector_at_coef
from linearize import LinearizedModelWrapper
from task_vectors import T5NonLinearTaskVector
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

if args.seed is not None:
        args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.model}"

accuracies = {}

TASKS = ["cola", "sst2", "mrpc", "rte"]

print("*" * 100)
print("Evaluating TIES-Merging models.")
ft_accuracies_path = os.path.join(args.save, "ft_accuracies_ties.json")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)


tokenizer = T5Tokenizer.from_pretrained(args.model)

eval_datasets = [
    "cola",
    "mrpc",
    "rte",
    "sst2",
]

task_vectors = []
sign_vectors = []

for dataset in eval_datasets:
    pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot"
    finetuned_checkpoint = f"{args.save}/{dataset}/finetuned"
    task_vectors = T5NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    # TIES-Merging
    if sparsity > 0.0: # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
        with torch.no_grad():
            # Trim redundant params (according to global magnitude)
            global_scores = torch.cat([torch.flatten(v).abs() for v in task_vector.vector.values()])
            threshold, _ = torch.kthvalue(global_scores, int(sparsity * global_scores.numel()))
            sgn = {}
            for key in task_vector.vector:
                score = task_vector.vector[key].abs()
                task_vector.vector[key].mul_(torch.where(score <= threshold, 0.0, 1.0))
                # Store signs
                sgn[key] = torch.sign(task_vector.vector[key])

    task_vectors.append(task_vector)
    sign_vectors.append(sgn)

with torch.no_grad():
    # Elect final sign
    agg_task_vector = {}
    for vect in task_vectors:
        for key in vect.vector:
            if key not in agg_task_vector:
                agg_task_vector[key] = vect.vector[key].clone()
            else:
                agg_task_vector[key] += vect.vector[key].clone()

    majority_sign = torch.sign(torch.cat([torch.flatten(v).abs() for v in agg_task_vector.values()]).sum())

    # Disjoint merge
    non_zero_counts = {}
    disjoint_agg = {}
    for vect in task_vectors:
        for key in vect.vector:
            sgn_m = torch.sign(agg_task_vector[key])
            sgn_m[sgn_m == 0] = majority_sign

            rows_to_keep = torch.where(sgn_m > 0, vect.vector[key] > 0, vect.vector[key] < 0)
            selected_entries = vect.vector[key] * rows_to_keep

            if key not in non_zero_counts:
                non_zero_counts[key] = (selected_entries != 0).float()
                disjoint_agg[key] = selected_entries
            else:
                non_zero_counts[key] += (selected_entries != 0).float()
                disjoint_agg[key] += selected_entries

    for key in non_zero_counts:
        disjoint_agg[key] /= torch.clamp(non_zero_counts[key], min=1)
    
    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    for key in task_vector.vector:
        task_vector.vector[key].copy_(disjoint_agg[key])

args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    tokenizer,
    args,
    posthoc_linearization=False,
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    tokenizer, 
    args,
    float(optimal_coef),
    posthoc_linearization=False,
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

save_file = f"{args.save}/ties_additions.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)


