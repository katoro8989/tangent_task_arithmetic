import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

import argparse
from eval import eval_single_dataset
from linearize import LinearizedModelWrapper
from task_vectors import LinearizedTaskVector, NonLinearTaskVector
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwarg
from torch.utils.data import DataLoader


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
    "cola",
    # "mnli",
    # "mrpc",
    # "qnli",
    # "qqp",
    # "rte",
    # "sst2",
    # "stsb",
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

    dataset_class = load_dataset("glue", args.task)
    
    preprocessor_class = get_preprocessor(args.task)
    preprocessor = preprocessor_class(tokenizer=tokenizer, tokenizer_kwargs=args.tokenizer_kwargs)
    map_kwargs = get_map_kwargs(args.task)
    encoded_dataset = dataset_class.map(preprocessor, **map_kwargs)

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
        eval_dataloader = DataLoader(encoded_dataset["validation_matched"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
    else:
        eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # Evaluate
    print("=" * 100)
    print(f"Evaluating on validation split.")
    accuracies[dataset] = eval_single_dataset(
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
