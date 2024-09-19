import os
import time
import uuid

import torch
import wandb

from src import utils  # utilsがtorch_saveとtorch_loadを含むと仮定
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder
from src.utils import LabelSmoothing, cosine_lr

class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs_per_task):
        # 入力を連結
        all_inputs = torch.cat(inputs_per_task, dim=0)
        # 一度だけ image_encoder を呼び出す
        all_features = self.image_encoder(all_inputs)
        # 特徴量をタスクごとに分割
        features_per_task = []
        start_idx = 0
        for inputs in inputs_per_task:
            end_idx = start_idx + inputs.size(0)
            features = all_features[start_idx:end_idx]
            features_per_task.append(features)
            start_idx = end_idx
        # 各分類ヘッドに特徴量を渡す
        outputs = []
        for idx, features in enumerate(features_per_task):
            logits = self.classification_heads[idx](features)
            outputs.append(logits)
        return outputs

def finetune(rank, args, group):
    setup_ddp(rank, args.world_size, port=args.port)

    run = wandb.init(
        config=vars(args),
        project=f"{args.model}_multitask_{args.finetuning_mode}_orth",
        entity='katoro13',
        name=f"process_{rank}",
        group=group,
    )

    train_datasets = args.train_datasets
    ckpdir = os.path.join(args.save, "multitask")

    assert args.finetuning_mode in [
        "linear",
        "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    # チェックポイントが既に存在する場合はスキップ
    ft_path = (
        os.path.join(ckpdir, f"linear_finetuned.pt")
        if linearized_finetuning
        else os.path.join(ckpdir, f"finetuned.pt")
    )
    zs_path = (
        os.path.join(ckpdir, f"linear_zeroshot.pt")
        if linearized_finetuning
        else os.path.join(ckpdir, f"zeroshot.pt")
    )
    if os.path.exists(zs_path) and os.path.exists(ft_path):
        print(f"Skipping fine-tuning because {ft_path} exists.")
        return zs_path, ft_path

    if args.load is not None and args.load.endswith("pt"):
        image_encoder = (
            LinearizedImageEncoder.load(args.load)
            if linearized_finetuning
            else ImageEncoder.load(args.load)
        )
    else:
        print("Building image encoder.")
        image_encoder = (
            LinearizedImageEncoder(args, keep_lang=False)
            if linearized_finetuning
            else ImageEncoder(args)
        )

    image_encoder = image_encoder.cuda()

    # 各タスクの分類ヘッドを作成
    classification_heads = []
    for dataset in train_datasets:
        classification_head = get_classification_head(args, dataset)
        classification_head = classification_head.cuda()
        classification_heads.append(classification_head)

    # マルチヘッドの画像分類器を作成
    model = MultiHeadImageClassifier(image_encoder, classification_heads)
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 10

    # すべてのデータセットのデータローダーを設定
    data_loaders = []
    for dataset in train_datasets:
        dataset_obj = get_dataset(
            dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        data_loader = get_dataloader(dataset_obj, is_train=True, args=args, image_encoder=None)
        data_loaders.append(data_loader)

    # データとモデルをGPUに分散
    ddp_loaders = []
    for data_loader in data_loaders:
        ddp_loader = distribute_loader(data_loader)
        ddp_loaders.append(ddp_loader)

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # 各データセットのバッチ数を取得
    num_batches_per_dataset = [len(loader) for loader in ddp_loaders]
    num_steps_per_epoch = max(num_batches_per_dataset) // args.num_grad_accumulation
    total_steps = args.epochs * num_steps_per_epoch

    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup_length,
        total_steps,
    )

    # ゼロショットモデルを保存
    if args.save is not None and is_main_process():
        os.makedirs(ckpdir, exist_ok=True)
        model_path = (
            os.path.join(ckpdir, f"linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"zeroshot.pt")
        )
        ddp_model.module.image_encoder.save(model_path)

    # 各データローダーのイテレータを作成
    ddp_loader_iters = [iter(loader) for loader in ddp_loaders]

    count_step = 0  # ステップカウンタを初期化
    max_steps = 2000  # 最大ステップ数を設定

    for epoch in range(args.epochs):
        if count_step >= max_steps:
            print(f"Reached maximum steps of {max_steps}. Ending training.")
            break  # 外側のループを終了
        ddp_model.train()

        for i in range(num_steps_per_epoch):
            if count_step >= max_steps:
                print(f"Reached maximum steps of {max_steps}. Ending training.")
                break  # 内側のループを終了

            start_time = time.time()
            step = count_step // args.num_grad_accumulation

            # inputs_per_task と labels_per_task を収集
            inputs_per_task = []
            labels_per_task = []
            data_time = 0.0

            for idx in range(len(train_datasets)):
                try:
                    batch = next(ddp_loader_iters[idx])
                except StopIteration:
                    ddp_loader_iters[idx] = iter(ddp_loaders[idx])
                    batch = next(ddp_loader_iters[idx])

                batch_time_start = time.time()
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].cuda()
                labels = batch["labels"].cuda()
                batch_time_end = time.time()

                data_time += batch_time_end - batch_time_start

                inputs_per_task.append(inputs)
                labels_per_task.append(labels)

            # モデルの forward を一度だけ呼び出す
            outputs = ddp_model(inputs_per_task)
            total_loss = 0.0
            accuracies = []

            for logits, labels in zip(outputs, labels_per_task):
                loss = loss_fn(logits, labels)
                total_loss += loss

                _, preds = torch.max(logits, 1)
                correct = torch.sum(preds == labels).item()
                accuracy = correct / labels.size(0)
                accuracies.append(accuracy)

            # 損失を正規化
            total_loss = total_loss / len(train_datasets)

            # backward を一度だけ呼び出す
            total_loss.backward()

            if (count_step + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time = time.time() - start_time

            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
                and is_main_process()
            ):
                print("Saving checkpoint.")
                model_path = (
                    os.path.join(ckpdir, f"linear_checkpoint_{step}.pt")
                    if linearized_finetuning
                    else os.path.join(ckpdir, f"checkpoint_{step}.pt")
                )
                ddp_model.module.save(model_path)

            if (
                step % print_every == 0
                and ((count_step + 1) % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / num_steps_per_epoch

                avg_accuracy = sum(accuracies) / len(accuracies)

                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_steps_per_epoch}]\t"
                    f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                    f"Avg Acc: {avg_accuracy:.4f}",
                    flush=True,
                )
                # 各タスクの精度をログ
                log_dict = {
                    'step': step,
                    'total_loss': total_loss.item(),
                    'avg_accuracy': avg_accuracy,
                }
                for idx, dataset in enumerate(train_datasets):
                    log_dict[f'acc_{dataset}'] = accuracies[idx]
                run.log(log_dict)

            count_step += 1  # ステップカウンタをインクリメント

    if args.save is not None and is_main_process():
        zs_path = (
            os.path.join(ckpdir, f"linear_zeroshot.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"zeroshot.pt")
        )
        ft_path = (
            os.path.join(ckpdir, f"linear_finetuned.pt")
            if linearized_finetuning
            else os.path.join(ckpdir, f"finetuned.pt")
        )
        ddp_model.module.save(ft_path)
        return zs_path, ft_path

    cleanup_ddp()


if __name__ == "__main__":
    train_datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]

    args = parse_arguments()

    args.lr = 1e-5
    args.epochs = 1000  # 必要に応じて調整してください
    args.train_datasets = [d + "Val" for d in train_datasets]

    # 勾配の累積を使用して大きなバッチサイズをシミュレート
    args.batch_size = 16 if args.model == "ViT-L-14" else 32
    args.num_grad_accumulation = 8 if args.model == "ViT-L-14" else 1

    if args.seed is not None:
        args.save = f"/mnt/data/checkpoints_ours_{args.seed}/{args.model}"
    else:
        args.save = f"/mnt/data/checkpoints_ours/{args.model}"
    print("=" * 100)
    print(f"Finetuning {args.model} on multiple datasets")
    print("=" * 100)

    group = "{}_{}".format(time.strftime('%Y%m%d-%H%M%S'), str(uuid.uuid4()))

    torch.multiprocessing.spawn(finetune, args=(args, group), nprocs=args.world_size)