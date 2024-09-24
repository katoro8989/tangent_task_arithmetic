import json

from src.args import parse_arguments
from src.eval import eval_dp_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
if args.seed is not None:
    args.save = f"/mnt/data2/checkpoints_ours_continual_pair{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data2/checkpoints_ours_continual_pair/{args.model}"

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

for check_point in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    accuracies[check_point] = {}
    eval_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    for task_a in eval_datasets:
        print("*" * 100)
        print(f"Evaluating on {task_a}")

        pretrained_checkpoint = (
            f"{args.save}/{task_a}Val/linear_zeroshot.pt"
            if args.finetuning_mode == "linear"
            else f"{args.save}/{task_a}Val/zeroshot.pt"
        )

        finetuned_checkpoint = (
            f"{args.save}/{task_a}Val/linear_checkpoint_{check_point}.pt"
            if args.finetuning_mode == "linear"
            else f"{args.save}/{task_a}Val/finetuned.pt"
        )

        try:
            task_vector = (
                LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
                if args.finetuning_mode == "linear"
                else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

        if args.finetuning_mode == "none":
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear":
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        elif args.finetuning_mode == "posthoc":
            zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
            ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
            image_encoder = LinearizedImageEncoder(
                init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
            )

        accuracies[check_point][task_a] = {}

        for task_b in eval_datasets:
            for split in ["test", "val"]:
                # Evaluate
                print("=" * 100)
                print(f"Evaluating on {split} split.")
                eval_dataset = task_b if split == "test" else f"{task_b}Val"

                accuracies[check_point][task_a][eval_dataset] = eval_dp_single_dataset(
                    image_encoder, eval_dataset, args
                )["dp_norm_ave"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_dp.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_dp.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_dp.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_dp.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
