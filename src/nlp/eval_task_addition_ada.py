import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import sys
from tqdm import tqdm
import torch
import gc
from task_vectors import T5NonLinearTaskVector
from eval import eval_single_dataset
import argparse
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from linearize import SimpleCallableHFModel


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ["cola", "sst2", "mrpc", "rte"]
model = "google/flan-t5-small"

parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default="google/flan-t5-small")
parser.add_argument('--output_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

args.model = model
args.seed = 42
args.data_dir = "/mnt2/dataset/glue_split"
args.device = "cuda" if torch.cuda.is_available() else "cpu"


if args.seed is not None:
    args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"/mnt2/t5_glue_checkpoints/{args.model}"

args.logs_path = args.save
pretrained_checkpoint = f"{args.save}/cola/zeroshot"

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [T5NonLinearTaskVector(pretrained_checkpoint, f"{args.save}/{dataset_name}/finetuned") for dataset_name in exam_datasets]

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)
   


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, input_ids, attention_mask, labels):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return features

class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        prior = 0.3
        rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, input_ids, attention_mask, labels, dataset_name):
        # alph = self.lambdas()
        # params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        # params = tuple(p.cuda(0) for p in params)

        # load_weights(self.model, self.names, params)
        # out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # torch.cuda.empty_cache()
        alph = self.lambdas()  # Shape: (1, num_sources)

        weights = alph.view(-1)  # Shape: (num_sources,)

        blended_params = []

        for name, param_group in zip(self.names, zip(*self.paramslist)):
            # param_group は各ソースからのテンソルのタプル
            param_shapes = [p.shape for p in param_group]
            if not all(shape == param_shapes[0] for shape in param_shapes):
                raise ValueError(f"Shape mismatch in parameter '{name}'. Expected all sources to have shape {param_shapes[0]}, but got {[s for s in param_shapes]}")

            stacked = torch.stack(param_group, dim=0)  # Shape: (num_sources, *param_shape)

            reshaped_weights = weights.view(-1, *([1] * (stacked.dim() - 1)))  # Shape: (num_sources, 1, 1, ...)

            blended = (stacked * reshaped_weights).sum(dim=0)  # Shape: same as individual parameter

            if blended.shape != self.model.state_dict()[name].shape:
                raise ValueError(f"Blended parameter '{name}' shape {blended.shape} does not match model parameter shape {self.model.state_dict()[name].shape}")

            blended_params.append(blended)

        # モデルのパラメータを直接更新（torch.no_grad()コンテキスト内で）
        with torch.no_grad():
            for name, blended in zip(self.names, blended_params):
                if name in self.model.state_dict():
                    self.model.state_dict()[name].copy_(blended.to(self.device))
                else:
                    print(f"Parameter '{name}' not found in the model's state_dict.")

        # フォワードパス
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        torch.cuda.empty_cache()

        return out

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

tokenizer = T5Tokenizer.from_pretrained(args.model)

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

hf_t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_checkpoint)
pretrained_model = SimpleCallableHFModel(hf_t5_model)
pretrained_model_dic = pretrained_model.state_dict()

model = ModelWrapper(pretrained_model, exam_datasets)
model = model.to(args.device)
_, names = make_functional(model)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
torch.cuda.empty_cache()
adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)

print('init lambda:')
print(adamerging_mtl_model.lambdas())
print('collect_trainable_params:')
print(list(adamerging_mtl_model.collect_trainable_params()))

epochs = 500
optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
accumulation_steps = 8

dataloaders = {}
for dataset_name in exam_datasets:
    #from args.data_dir/dataset_name
    task_dir = os.path.join(args.data_dir, dataset_name)
    encoded_dataset = load_from_disk(task_dir)
    dataloader = DataLoader(encoded_dataset["test"], batch_size=2, collate_fn=collate_fn, shuffle=True)
    dataloaders[dataset_name] = dataloader
data_iters = {dataset_name: iter(dataloaders[dataset_name]) for dataset_name in exam_datasets}

def log_memory_usage(stage):
    print(f"[{stage}] Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"[{stage}] Memory Reserved: {torch.cuda.memory_reserved() / 1e6} MB")

for epoch in tqdm(range(epochs), desc="Training"):
    log_memory_usage("Start of Epoch")
    losses = 0.
    for dataset_name in exam_datasets:
        log_memory_usage(f"Before {dataset_name} Forward Pass")
        try:
            data_iter = data_iters[dataset_name]
            data = next(data_iter)
        except StopIteration:
            dataloader = dataloaders[dataset_name]
            data_iters[dataset_name] = iter(dataloader)
            data_iter = data_iters[dataset_name]
            data = next(data_iter)
        log_memory_usage(f"After Getting Data")
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        labels = data['labels'].to(args.device)

        log_memory_usage(f"Before Prediction")
        outputs = adamerging_mtl_model(input_ids, attention_mask, labels, dataset_name)
        log_memory_usage(f"After Prediction")
        loss = softmax_entropy(outputs).mean(0)
        losses += loss

        log_memory_usage(f"After {dataset_name} Forward Pass")

        # 不要なテンソルの削除
        del x, y, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()


    optimizer.zero_grad()
    losses.backward()
    optimizer.step()


    if ((epoch+1) % 500) == 0:
        log.info(str(list(adamerging_mtl_model.lambdas().data)))

        Total_ACC = 0.
        for dataset_name in exam_datasets:
            model = adamerging_mtl_model.get_image_encoder()
            eval_dataloader = dataloaders[dataset_name]
            metrics = eval_single_dataset(model, tokenizer, eval_dataloader, args)
            Total_ACC += metrics['top1']
            log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
        log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')