import json
import os
import itertools

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()

if args.seed is not None:
    args.save = f"/mnt/data/checkpoints_ours_{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data/checkpoints_ours/{args.model}"


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
            args.task_to_orth = pair_dataset[1 - i]
            pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
            # finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned_orth_to_{args.task_to_orth}.pt"
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

    args.eval_datasets = [dataset + "Val" for dataset in pair_dataset]
    args.control_dataset = None

    # We use the validation set to choose the optimal coefficient.
    val_metrics = evaluate_task_vector(
        task_vector,
        pretrained_checkpoint,
        args,
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric="avg_normalized_top1",
        minimize=False,
    )

    # Evaluate on the test set with the optimal coefficient.
    args.eval_datasets = [dataset for dataset in pair_dataset]
    test_metrics = evaluate_task_vector_at_coef(
        task_vector,
        pretrained_checkpoint,
        args,
        float(optimal_coef),
        posthoc_linearization=args.finetuning_mode == "posthoc",
    )

    print("=" * 100)
    print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
    print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
    additive_accuracies = {"test": test_metrics, "val": val_metrics}
    final_additive_accuracies[f"{pair_dataset[0]}-{pair_dataset[1]}"] = additive_accuracies

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions_2pair.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions_2pair.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions_2pair.json"
with open(save_file, "w") as f:
    json.dump(final_additive_accuracies, f, indent=4)
