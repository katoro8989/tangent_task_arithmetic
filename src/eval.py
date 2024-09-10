import numpy as np
import torch
import tqdm

from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageClassifier


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    torch.cuda.set_device(0)
    args.batch_size = 16

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")

    return metrics

def eval_dp_single_dataset(image_encoder, dataset_name, args):
    model = image_encoder

    model.eval()
    torch.cuda.set_device(3)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        norm_mean_total = 0.
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            dps = utils.get_dps(x, model)
            dp_norms = torch.norm(dps, dim=1)  # 各 dp の L2 ノルム (2-ノルム) を計算
            norm_mean_batch = dp_norms.sum()  # ノルムの和を計算
            norm_mean_total += norm_mean_batch.item()
            
    metrics = {"dp_norm_ave": norm_mean_total / len(dataset.test_dataset)}
    print(f"Done evaluating on {dataset_name}. dp_norm_ave: {norm_mean_total / len(dataset.test_dataset)}")

    return metrics

def eval_weight_disentanglement_single_dataset(
        image_encoder_alone, image_encoder_both, dataset_name, args
        ):
    
    torch.cuda.set_device(0)
    args.batch_size = 16

    classification_head = get_classification_head(args, dataset_name)
    model_alone = ImageClassifier(image_encoder_alone, classification_head)
    model_both = ImageClassifier(image_encoder_both, classification_head)

    model_alone.eval()
    model_both.eval()

    dataset = get_dataset(
        dataset_name,
        model_alone.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"]

            logits_alone = utils.get_logits(x, model_alone)
            logits_both = utils.get_logits(x, model_both)

            pred_alone = logits_alone.argmax(dim=1, keepdim=True).to(device)
            pred_both = logits_both.argmax(dim=1, keepdim=True).to(device)

            correct += pred_alone.eq(pred_both).sum().item()

            n += y.size(0)

        disentanglement = correct / n

    return disentanglement


def evaluate(image_encoder, args):
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

        results = eval_single_dataset(image_encoder, dataset_name, args)

        print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        per_dataset_results[dataset_name + ":top1"] = results["top1"]

    return per_dataset_results


def evaluate_task_vector_at_coef(
    task_vector, pretrained_checkpoint, args, scaling_coef, posthoc_linearization=False
):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    if posthoc_linearization:
        pretrained_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )
        image_encoder = LinearizedImageEncoder(
            init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
        )
    coef_info = evaluate(image_encoder, args)

    coef_info = add_normalized_accuracy(coef_info, args)
    coef_info["avg_normalized_top1"] = np.mean(
        [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
    )
    coef_info["avg_top1"] = np.mean(
        [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
    )

    return coef_info


def evaluate_task_vector(
    task_vector, pretrained_checkpoint, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
        print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
        info[scaling_coef] = evaluate_task_vector_at_coef(
            task_vector,
            pretrained_checkpoint,
            args,
            scaling_coef,
            posthoc_linearization,
        )

    return info

def evaluate_weight_disentanglement_at_coefs(
    task_vector1, task_vector2, pretrained_checkpoint, coef1, coef2, args
):
    eval_datasets = args.eval_datasets
    task_vector_to_evaluate_1 = coef1 * task_vector1
    task_vector_to_evaluate_2 = coef2 * task_vector2
    task_vector_to_evaluate_both = coef1 * task_vector1 + coef2 * task_vector2

    image_encoder_1 = task_vector_to_evaluate_1.apply_to(
        pretrained_checkpoint, scaling_coef=1.
    )
    image_encoder_2 = task_vector_to_evaluate_2.apply_to(
        pretrained_checkpoint, scaling_coef=1.
    )
    image_encoder_both = task_vector_to_evaluate_both.apply_to(
        pretrained_checkpoint, scaling_coef=1.
    )

    disentanglement_1 = eval_weight_disentanglement_single_dataset(
        image_encoder_1, image_encoder_both, eval_datasets[0], args
    )

    disentanglement_2 = eval_weight_disentanglement_single_dataset(
        image_encoder_2, image_encoder_both, eval_datasets[1], args
    )

    metric = {"disentanglement": (disentanglement_1 + disentanglement_2) / 2}
    return metric


def evaluate_weight_disentanglement(
    task_vector1, task_vector2, pretrained_checkpoint, args, posthoc_linearization=False
):
    info = {}
    for scaling_coef1 in np.linspace(-3., 3., args.n_eval_points):
        for scaling_coef2 in np.linspace(-3., 3., args.n_eval_points):
            print(f"Evaluating for scaling coefficient {scaling_coef1:.2f}-{scaling_coef2:.2f}")
            info[scaling_coef1][scaling_coef2] = evaluate_weight_disentanglement_at_coefs(
                task_vector1, 
                task_vector2, 
                pretrained_checkpoint,
                scaling_coef1, 
                scaling_coef2, 
                args,
            )

    return info


def add_normalized_accuracy(results, args):
    for dataset_name in args.eval_datasets:
        results[dataset_name + ":normalized_top1"] = (
            results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
        )

    return results


def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
    err_linear = 1 - acc_linear
    err_nonlinear = 1 - acc_nonlinear
    return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)
