import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_from_disk
import torch


import argparse
from eval import eval_single_dataset
from linearize import LinearizedModelWrapper
from task_vectors import T5NonLinearTaskVector
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="google/flan-t5-small")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
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

accuracies = {}


print("*" * 100)
print("Evaluating TIES-Merging models.")

for sparsity in [0.9, 0.95, 0.99]:
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
            f"{args.save}/{dataset}/zeroshot"
        )

        finetuned_checkpoint = (
            f"{args.save}/{dataset}/finetuned"
        )

        try:
            task_vector = (
                T5NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            )
        except FileNotFoundError:
            print(f"Error: Could not find {finetuned_checkpoint}.")
            continue

        # TIES-Merging
        if (
            sparsity > 0.0
        ):  # NOTE: if sparsity == 0.0 we have the standard non-linear finetuning
            with torch.no_grad():
                global_scores = torch.cat(
                    [torch.flatten(v).abs() for v in task_vector.vector.values()]
                )
                threshold, _ = torch.kthvalue(
                    global_scores, int(sparsity * global_scores.numel())
                )
                for key in task_vector.vector:
                    if any(x in key for x in ["attn", "mlp", "conv"]):
                        # Trim redundant params (according to global magnitude)
                        score = task_vector.vector[key].abs()
                        task_vector.vector[key].mul_(
                            torch.where(score <= threshold, 0.0, 1.0)
                        )

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
            accuracies[eval_dataset + f"_{sparsity}"] = eval_single_dataset(
                model, tokenizer, eval_dataloader, args
            )["top1"]


# Save results

save_path = f"{args.save}/ft_accuracies_ties.json"

with open(save_path, "w") as f:
    json.dump(accuracies, f)
