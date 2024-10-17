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
from torch.utils.data import DataLoader


from linearize import LinearizedModel, LinearizedModelWraper, SimpleCallableT5Model

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

    

    for name, param in model_class.named_parameters():
        print(name, param.shape)

    simple_model_class = SimpleCallableT5Model(model_class)

    for name, param in simple_model_class.named_parameters():
        print(name, param.shape)
    
    model_class = LinearizedModelWraper(simple_model_class)

    sample = encoded_dataset["train"][0]  # "train"データセットの最初のサンプル

    # サンプル内容を表示
    print("Sample input_ids:", sample["input_ids"])
    print("Sample attention_mask:", sample["attention_mask"])
    print("Sample labels:", sample["labels"])

    # input_ids と attention_mask をリストからテンソルに変換
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0)  # リストをテンソルに変換してバッチ次元を追加
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)  # 同様にテンソルに変換してバッチ次元を追加
    labels = torch.tensor(sample["labels"]).unsqueeze(0)  # ラベルも同様にテンソルに変換


    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # サンプルデータをデバイスに転送
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # モデルもデバイスに転送
    model = model_class.to(device)

    # モデルにサンプルデータを入力し、出力を取得
    model.eval()  # 評価モードにする
    with torch.no_grad():  # 勾配計算を無効化
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # モデルの出力（ロジットや損失）を取得
    logits = outputs  # 生の出力 (logits)

    print("logit", logits)

    # ロジットから最も高い確率のトークンIDを取得
    predicted_ids = torch.argmax(logits, dim=-1)

    # トークンIDをデコードしてテキストとして表示
    predicted_text = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    print("Predicted text:", predicted_text)

    # オリジナルのテキストも確認
    decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print("Original input text:", decoded_input)

    


    


    # # DataLoaderの作成
    # train_dataloader = DataLoader(encoded_dataset["train"], batch_size=args.train_batch_size, shuffle=True)
    # eval_dataloader = DataLoader(encoded_dataset["validation"], batch_size=args.eval_batch_size)

    
    # # Distribute the data and model across the GPUs.
    # ddp_loader = distribute_loader(data_loader)
    # ddp_model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[rank],
    #     find_unused_parameters=True,
    #     output_device=rank,
    # )

    # if args.ls > 0:
    #     loss_fn = LabelSmoothing(args.ls)
    # else:
    #     loss_fn = torch.nn.CrossEntropyLoss()

    # params = [p for p in ddp_model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # scheduler = cosine_lr(
    #     optimizer,
    #     args.lr,
    #     args.warmup_length,
    #     args.epochs * num_batches // args.num_grad_accumulation,
    # )

    # # Saving zero-shot model
    # if args.save is not None and is_main_process():
    #     os.makedirs(ckpdir, exist_ok=True)
    #     model_path = (
    #         os.path.join(ckpdir, "linear_zeroshot.pt")
    #         if linearized_finetuning
    #         else os.path.join(ckpdir, "zeroshot.pt")
    #     )
    #     ddp_model.module.image_encoder.save(model_path)

    # for epoch in range(args.epochs):
    #     ddp_model.train()

    #     for i, batch in enumerate(ddp_loader):
    #         start_time = time.time()

    #         step = (
    #             i // args.num_grad_accumulation
    #             + epoch * num_batches // args.num_grad_accumulation
    #         )

    #         batch = maybe_dictionarize(batch)
    #         inputs = batch["images"].cuda()
    #         labels = batch["labels"].cuda()
    #         data_time = time.time() - start_time

    #         logits = ddp_model(inputs)

    #         loss = loss_fn(logits, labels)

    #         loss.backward()

    #         if (i + 1) % args.num_grad_accumulation == 0:
    #             scheduler(step)

    #             torch.nn.utils.clip_grad_norm_(params, 1.0)
    #             optimizer.step()
    #             optimizer.zero_grad()

    #         batch_time = time.time() - start_time

    #         if (
    #             args.checkpoint_every > 0
    #             and step % args.checkpoint_every == 0
    #             and is_main_process()
    #         ):
    #             print("Saving checkpoint.")
    #             model_path = (
    #                 os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
    #                 if linearized_finetuning
    #                 else os.path.join(ckpdir, f"checkpoint_{step}.pt")
    #             )
    #             ddp_model.module.image_encoder.save(model_path)

    #         if (
    #             step % print_every == 0
    #             and ((i + 1) % args.num_grad_accumulation == 0)
    #             and is_main_process()
    #         ):
    #             percent_complete = 100 * i / len(ddp_loader)

    #             _, preds = torch.max(logits, 1)
    #             correct = torch.sum(preds == labels).item()
    #             accuracy = correct / labels.size(0)

    #             print(
    #                 f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"  # noqa: E501
    #                 f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
    #                 f"Acc: {accuracy}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",  # noqa: E501
    #                 flush=True,
    #             )
    #             run.log({
    #                 'step': step,
    #                 'train_loss': loss.item(),
    #                 'train_accuracy': accuracy, 
    #             })

    # # FIXME: Make this work with DDP.
    # if is_main_process():
    #     # We only need to evaluate the model on the first GPU.
    #     image_encoder = ddp_model.module.image_encoder
    #     eval_single_dataset(image_encoder, train_dataset, args)

    # if args.save is not None and is_main_process():
    #     zs_path = (
    #         os.path.join(ckpdir, "linear_zeroshot.pt")
    #         if linearized_finetuning
    #         else os.path.join(ckpdir, "zeroshot.pt")
    #     )
    #     ft_path = (
    #         os.path.join(ckpdir, "linear_finetuned.pt")
    #         if linearized_finetuning
    #         else os.path.join(ckpdir, "finetuned.pt")
    #     )
    #     image_encoder.save(ft_path)
    #     return zs_path, ft_path

    # cleanup_ddp()


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