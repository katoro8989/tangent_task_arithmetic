import os
import time
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, load_metric
from dataset_preprocess.glue_process import *
import argparse
import datetime
import wandb
import evaluate

from linearize import LinearizedModel, LinearizedModelWraper

preprocessor_mapping = {
    "cola": CoLA_Preprocessor,
    "rte": RTE_Preprocessor,
    "mnli": MNLI_Preprocessor,
    "mrpc": MRPC_Preprocessor,
    "qnli": QNLI_Preprocessor,
    "qqp": QQP_Preprocessor,
    "sst2": SST2_Preprocessor,
    "stsb": STSB_Preprocessor,
}

tokenizer_kwargs = {
  "padding": "max_length",
  "truncation": True,
  "return_tensors": "pt",
}

map_kwargs = {
    "remove_columns": ["sentence", "label", "idx"],
    "batched": True,
    "num_proc": 1,
    "desc": "Running tokenizer on dataset"
}

def finetune(args):
    
    run_id = f'{args.model}'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f'{run_id}_{timestamp}'
    output_dir = os.path.join(args.output_dir, run_id)

    
    model_class = T5ForConditionalGeneration.from_pretrained(args.model)
    
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    dataset_class = load_dataset("glue", args.task)
    
    preprocessor_class = preprocessor_mapping[args.task]
    preprocessor = preprocessor_class(tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs)
    encoded_dataset = dataset_class.map(preprocessor, **map_kwargs)
    
    if args.wandb:
        report = "wandb"
        print("=====wandb logging starts=====")
        wandb.init(project="flan-t5_glue",
            name=run_id,
            group="katoro13")
    else:
        report = None
    
    training_args = TrainingArguments(
        output_dir=output_dir,          
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        learning_rate=args.lr,        
        per_device_train_batch_size=args.train_batch_size,  
        per_device_eval_batch_size=args.eval_batch_size,   
        warmup_steps=args.warmup_steps,                
        weight_decay=args.weight_decay,               
        logging_strategy=args.logging_strategy, 
        run_name=run_id,  
        report_to=report, 
        fp16=args.fp16, 
        logging_dir='./logs', 
        evaluation_strategy=args.evaluation_strategy, 
        fp16_full_eval=args.fp16_full_eval, 
        eval_steps=args.eval_steps, 
        eval_accumulation_steps=args.eval_accumulation_steps, 
        auto_find_batch_size=args.auto_find_batch_size, 
    )

    
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        predictions = [pred if pred in ['0', '1'] else '2' for pred in predictions] 
        
        # predictions = np.array([int(pred) for pred in predictions])
        # labels = np.array([int(label) for label in labels])

        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        # correct_predictions = [pred == label for pred, label in zip(predictions, labels)]
        # accuracy = sum(correct_predictions) / len(correct_predictions)
        # return accuracy
        
        return metric.compute(predictions=predictions, references=labels)

    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    
    trainer = Trainer(
        model=model_class,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        compute_metrics=lambda p: metric.compute(predictions=torch.argmax(p.predictions, axis=1), references=p.label_ids),
        tokenizer=tokenizer,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    trainer.train()

    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetuning of T5')
    parser.add_argument('--task', type=str, default="cola")
    parser.add_argument('--model', type=str, default="google/flan-t5-small")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=int, default=0.01)
    parser.add_argument('--fp16', action='store_true', help='whether fp16')
    parser.add_argument('--logging_dir', type=int, default=None)
    parser.add_argument('--logging_strategy', type=str, default="steps")
    parser.add_argument('--evaluation_strategy', type=str, default="steps")
    parser.add_argument('--wandb', action='store_true', help='whether log on wandb')
    parser.add_argument('--exp_id', type=str, default=None, help='exp id for reporting')
    parser.add_argument('--fp16_full_eval', action='store_true')
    parser.add_argument('--eval_steps', type=float, default=1000)
    parser.add_argument('--eval_accumulation_steps', type=int, default=10)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--auto_find_batch_size', action='store_true')
    args = parser.parse_args()
                    
    
    finetune(args)