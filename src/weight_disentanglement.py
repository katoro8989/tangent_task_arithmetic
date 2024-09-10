import json
import os
import itertools

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_weight_disentanglement
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"/mnt/data/checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data/checkpoints/{args.model}"


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
    "Cars",
    "DTD",
    # "EuroSAT",
    # "GTSRB",
    # "MNIST",
    # "RESISC45",
    # "SVHN",
    # "SUN397",
]

all_combinations = list(itertools.combinations(eval_datasets, 2))
final_weight_disentanglement = {}
for pair_dataset in all_combinations:

    task_vectors = []

    for i, dataset in enumerate(pair_dataset):
        if args.finetuning_mode == "linear":
            args.task_to_orth = pair_dataset[1 - i]
            if args.task_to_orth == "Cars":
                args.task_to_orth = args.task_to_orth + "Val"
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            # finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned_orth_to_{args.task_to_orth}.pt"
            task_vectors.append(
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
            task_vectors.append(
                NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )

    args.eval_datasets = [dataset for dataset in pair_dataset]
    args.control_dataset = None

    # We use the validation set to choose the optimal coefficient.
    test_metrics = evaluate_weight_disentanglement(
        task_vectors[0],
        task_vectors[1],
        pretrained_checkpoint,
        args,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    weight_disentanglement = {"test": test_metrics}
    final_weight_disentanglement[f"{pair_dataset[0]}-{pair_dataset[1]}"] = weight_disentanglement

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/weight_disentanglement.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/weight_disentanglement.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/weight_disentanglement.json"
with open(save_file, "w") as f:
    json.dump(final_weight_disentanglement, f, indent=4)
