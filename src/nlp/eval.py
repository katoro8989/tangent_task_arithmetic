import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# from src import utils
# from src.datasets.common import get_dataloader, maybe_dictionarize
# from src.datasets.registry import get_dataset
# from src.heads import get_classification_head
# from src.linearize import LinearizedImageEncoder
# from src.modeling import ImageClassifier


def eval_single_dataset(model, tokenizer, eval_dataloader, args):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs

            preds = torch.argmax(logits, dim=-1)

            predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if args.task == "cola":
        result = matthews_corrcoef(all_labels, all_preds)
    elif args.task == "mrpc" or args.tasf == "qqp":
        _result1 = accuracy_score(all_labels, all_preds)
        _result2 = f1_score(all_labels, all_preds)
        result = (_result1, _result2) / 2
    elif args.task == "stsb":
        _result1 = pearsonr(all_labels, all_preds)
        _result2 = spearmanr(all_labels, all_preds)
        result = (_result1, _result2) / 2
    else:
        result = accuracy_score(all_labels, all_preds)

    print(f"Done evaluating on {args.task}. Score: {100*result:.2f}%")

    return {"top1": result}

# def eval_weight_disentanglement_single_dataset(
#         image_encoder_alone, image_encoder_both, dataset_name, args
#         ):
    
#     torch.cuda.set_device(args.device_number)
#     args.batch_size = 128

#     classification_head = get_classification_head(args, dataset_name)
#     model_alone = ImageClassifier(image_encoder_alone, classification_head)
#     model_both = ImageClassifier(image_encoder_both, classification_head)

#     model_alone.eval()
#     model_both.eval()

#     dataset = get_dataset(
#         dataset_name,
#         model_alone.val_preprocess,
#         location=args.data_location,
#         batch_size=args.batch_size,
#     )
#     dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
#     device = args.device

#     with torch.no_grad():
#         top1, correct, n = 0.0, 0.0, 0.0
#         for _, data in enumerate(tqdm.tqdm(dataloader)):
#             data = maybe_dictionarize(data)
#             x = data["images"].to(device)
#             y = data["labels"]

#             logits_alone = utils.get_logits(x, model_alone)
#             logits_both = utils.get_logits(x, model_both)

#             pred_alone = logits_alone.argmax(dim=1, keepdim=True).to(device)
#             pred_both = logits_both.argmax(dim=1, keepdim=True).to(device)

#             correct += pred_alone.eq(pred_both).sum().item()

#             n += y.size(0)

#         disentanglement = correct / n

#     return disentanglement


# def evaluate(image_encoder, args):
#     if args.eval_datasets is None:
#         return
#     per_dataset_results = {}
#     eval_datasets = (
#         args.eval_datasets
#         if args.control_dataset is None
#         else args.eval_datasets + [args.control_dataset]
#     )
#     for dataset_name in eval_datasets:
#         print("Evaluating on", dataset_name)

#         results = eval_single_dataset(image_encoder, dataset_name, args)

#         print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
#         per_dataset_results[dataset_name + ":top1"] = results["top1"]

#     return per_dataset_results


# def evaluate_task_vector_at_coef(
#     task_vector, pretrained_checkpoint, args, scaling_coef, posthoc_linearization=False
# ):
#     image_encoder = task_vector.apply_to(
#         pretrained_checkpoint, scaling_coef=scaling_coef
#     )
#     if posthoc_linearization:
#         pretrained_encoder = task_vector.apply_to(
#             pretrained_checkpoint, scaling_coef=0.0
#         )
#         image_encoder = LinearizedImageEncoder(
#             init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
#         )
#     coef_info = evaluate(image_encoder, args)

#     coef_info = add_normalized_accuracy(coef_info, args)
#     coef_info["avg_normalized_top1"] = np.mean(
#         [coef_info[dataset + ":normalized_top1"] for dataset in args.eval_datasets]
#     )
#     coef_info["avg_top1"] = np.mean(
#         [coef_info[dataset + ":top1"] for dataset in args.eval_datasets]
#     )

#     return coef_info


# def evaluate_task_vector(
#     task_vector, pretrained_checkpoint, args, posthoc_linearization=False
# ):
#     info = {}
#     for scaling_coef in np.linspace(0.0, 1.0, args.n_eval_points):
#         print(f"Evaluating for scaling coefficient {scaling_coef:.2f}")
#         info[scaling_coef] = evaluate_task_vector_at_coef(
#             task_vector,
#             pretrained_checkpoint,
#             args,
#             scaling_coef,
#             posthoc_linearization,
#         )

#     return info

# def evaluate_weight_disentanglement_at_coefs(
#     task_vector1, task_vector2, pretrained_checkpoint, coef1, coef2, args
# ):
#     eval_datasets = args.eval_datasets
#     task_vector_to_evaluate_1 = task_vector1 * coef1
#     task_vector_to_evaluate_2 = task_vector2 * coef2
#     task_vector_to_evaluate_both = task_vector1 * coef1 + task_vector2 * coef2

#     image_encoder_1 = task_vector_to_evaluate_1.apply_to(
#         pretrained_checkpoint, scaling_coef=1.
#     )
#     image_encoder_2 = task_vector_to_evaluate_2.apply_to(
#         pretrained_checkpoint, scaling_coef=1.
#     )
#     image_encoder_both = task_vector_to_evaluate_both.apply_to(
#         pretrained_checkpoint, scaling_coef=1.
#     )

#     disentanglement_1 = eval_weight_disentanglement_single_dataset(
#         image_encoder_1, image_encoder_both, eval_datasets[0], args
#     )

#     disentanglement_2 = eval_weight_disentanglement_single_dataset(
#         image_encoder_2, image_encoder_both, eval_datasets[1], args
#     )

#     metric = {"disentanglement": (disentanglement_1 + disentanglement_2) / 2}
#     return metric


# def evaluate_weight_disentanglement(
#     task_vector1, task_vector2, pretrained_checkpoint, args, posthoc_linearization=False
# ):
#     info = {}
#     for scaling_coef1 in np.linspace(-3., 3., args.n_eval_points):
#         info[scaling_coef1] = {}
#         for scaling_coef2 in np.linspace(-3., 3., args.n_eval_points):
#             print(f"Evaluating for scaling coefficient {scaling_coef1:.2f}-{scaling_coef2:.2f}")
#             info[scaling_coef1][scaling_coef2] = evaluate_weight_disentanglement_at_coefs(
#                 task_vector1, 
#                 task_vector2, 
#                 pretrained_checkpoint,
#                 scaling_coef1, 
#                 scaling_coef2, 
#                 args,
#             )

#     return info


# def add_normalized_accuracy(results, args):
#     for dataset_name in args.eval_datasets:
#         results[dataset_name + ":normalized_top1"] = (
#             results[dataset_name + ":top1"] / args.finetuning_accuracies[dataset_name]
#         )

#     return results


# def nonlinear_advantage(acc_linear, acc_nonlinear, num_classes):
#     err_linear = 1 - acc_linear
#     err_nonlinear = 1 - acc_nonlinear
#     return (err_linear - err_nonlinear) * num_classes / (num_classes - 1)
