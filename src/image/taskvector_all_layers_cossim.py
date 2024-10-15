import json
import os
import itertools
import numpy as np

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"/mnt/data/checkpoints_ours_{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data/checkpoints_ours/{args.model}"


eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
]

all_combinations = list(itertools.combinations(eval_datasets, 2))
final_additive_accuracies = {}
for pair_dataset in all_combinations:

    task_vectors = []

    for i, dataset in enumerate(pair_dataset):
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors.append(
                NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )

    def cosine_similarity(weights1, weights2):
        """Compute the cosine similarity between two weight arrays."""
        dot_product = np.dot(weights1, weights2)
        norm_weights1 = np.linalg.norm(weights1)
        norm_weights2 = np.linalg.norm(weights2)
        return dot_product / (norm_weights1 * norm_weights2)

    def model_all_weights_similarity(vector_a, vector_b):
        """Compute the cosine similarity and Frobenius norm for all model weights combined."""
        vector_a_dict = vector_a.vector
        vector_b_dict = vector_b.vector

        all_weights_a = np.concatenate([w.flatten() for w in vector_a_dict.values()])
        all_weights_b = np.concatenate([w.flatten() for w in vector_b_dict.values()])

        cos_sim = cosine_similarity(all_weights_a, all_weights_b)

        return cos_sim
    
    cossim_result = model_all_weights_similarity(task_vectors[0], task_vectors[1])
    print(f"{pair_dataset[0]}-{pair_dataset[1]}: {cossim_result}")
    
    final_additive_accuracies[f"{pair_dataset[0]}-{pair_dataset[1]}"] = float(cossim_result)

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/cossim.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_cossim.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_cossim.json"
with open(save_file, "w") as f:
    json.dump(final_additive_accuracies, f, indent=4)
