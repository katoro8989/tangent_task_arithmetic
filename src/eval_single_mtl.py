import json

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
if args.seed is not None:
    args.save = f"/mnt/data/checkpoints_ours_{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data/checkpoints_ours/{args.model}"

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
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SUN397",
    "SVHN",
]:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    pretrained_checkpoint = (
        f"{args.save}/multitask/linear_zeroshot.pt"
        if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
        else f"{args.save}/multitask/zeroshot.pt"
    )

    finetuned_checkpoint = (
        f"{args.save}/multitask/linear_finetuned.pt"
        if args.finetuning_mode == "linear" or args.finetuning_mode == "none"
        else f"{args.save}/multitask/finetuned.pt"
    )

    # finetuned_checkpoint = (
    #     f"{args.save}/{dataset}Val/linear_finetuned_orth_to_{args.task_to_orth}.pt"
    #     if args.finetuning_mode == "linear"
    #     else f"{args.save}/{dataset}Val/finetuned_orth_to_{args.task_to_orth}.pt"
    # )

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
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
    elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear":
        image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    elif args.finetuning_mode == "posthoc":
        zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
        image_encoder = LinearizedImageEncoder(
            init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
        )

    for split in ["test", "val"]:
        # Evaluate
        print("=" * 100)
        print(f"Evaluating on {split} split.")
        eval_dataset = dataset if split == "test" else f"{dataset}Val"

        accuracies[eval_dataset] = eval_single_dataset(
            image_encoder, eval_dataset, args
        )["top1"]


if args.finetuning_mode == "none":
    # Evaluate zero-shot accuracy on ImageNet
    for split in ["ImageNetVal", "ImageNet"]:
        accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies_mtl.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies_mtl.json"
elif args.finetuning_mode == "linear":
    save_path = f"{args.save}/linear_ft_accuracies_mtl.json"
    # save_path = f"{args.save}/linear_ft_accuracies_cars_dtd.json"
elif args.finetuning_mode == "posthoc":
    save_path = f"{args.save}/posthoc_ft_accuracies_mtl.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
