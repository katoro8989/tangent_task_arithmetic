import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

import os
import sys


from datasets import load_from_disk
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils import collate_fn

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from src import utils
# from src.datasets.common import get_dataloader, maybe_dictionarize
# from src.datasets.registry import get_dataset
# from src.heads import get_classification_head
# from src.linearize import LinearizedImageEncoder
# from src.modeling import ImageClassifier


def eval_single_dataset(model, tokenizer, eval_dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device_number)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs

            preds = torch.argmax(logits, dim=-1)

            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]

            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Convert decoded predictions and labels to numpy arrays
            all_preds.extend(predictions)
            all_labels.extend(labels)

    if args.task == "cola":
        result = matthews_corrcoef(all_labels, all_preds)
    elif args.task == "mrpc" or args.task == "qqp":
        _result1 = accuracy_score(all_labels, all_preds)
        _result2 = f1_score(all_labels, all_preds, average='micro')
        result = (_result1 + _result2) / 2
    elif args.task == "stsb":
        _result1 = pearsonr(all_labels, all_preds)
        _result2 = spearmanr(all_labels, all_preds)
        result = (_result1 + _result2) / 2
    else:
        result = accuracy_score(all_labels, all_preds)

    print(f"Done evaluating on {args.task}. Score: {100*result:.2f}%")

    return {"top1": result}

def evaluate(model, tokenizer, args):
    if args.eval_datasets is None:
        return
    per_dataset_results = {}
    eval_datasets = (
        args.eval_datasets
        if args.control_dataset is None
        else args.eval_datasets + [args.control_dataset]
    )
    for dataset_name in eval_datasets:
        print("Evaluating on", dataset_name)
        if "Val" in dataset_name:
            args.task = dataset_name[:-3]
        else:
            args.task = dataset_name

        #from args.data_dir/dataset_name
        task_dir = os.path.join(args.data_dir, args.task)
        encoded_dataset = load_from_disk(task_dir)
        if "Val" in dataset_name:
            eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size, collate_fn=collate_fn)
        else:
            eval_dataloader = DataLoader(encoded_dataset["test"], batch_size=args.eval_batch_size, collate_fn=collate_fn)

        results = eval_single_dataset(model, tokenizer, eval_dataloader, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results

def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, tokenizer, args, scaling_coef, posthoc_linearization=False
):
    model = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    coef_info = evaluate(model, tokenizer, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, tokenizer, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef in np.linspace(0.0, args.eval_max_points, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            tokenizer, 
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results