import os
import time
import gc
import torch
import argparse
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from linearize import SimpleCallableHFModel
from task_vectors import T5NonLinearTaskVector
from eval import eval_single_dataset

# Utility to create log directory
def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Datasets and model configuration
exam_datasets = ["cola", "sst2", "mrpc", "rte"]
model_name = "google/flan-t5-small"

# Argument parser
parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--model', type=str, default=model_name)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

# Setup environment and paths
args.device = "cuda" if torch.cuda.is_available() else "cpu"
args.data_dir = "/mnt2/dataset/glue_split"
args.save = f"/mnt2/t5_glue_checkpoints_{args.seed}/{args.model}" if args.seed else f"/mnt2/t5_glue_checkpoints/{args.model}"
args.logs_path = args.save
pretrained_checkpoint = os.path.join(args.save, "cola/zeroshot")
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
args.log = create_log_dir(args.logs_path, f'log_{str_time_}_Task_wise_AdaMerging.txt')

# Load task vectors
task_vectors = [T5NonLinearTaskVector(pretrained_checkpoint, f"{args.save}/{dataset}/finetuned") for dataset in exam_datasets]

# Helper functions for model manipulation
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

# Wrapper class for the model
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Adaptive merging class
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        self.lambdas_raw = torch.nn.Parameter(torch.ones(1, len(paramslist) - 1) * 0.3)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, 0.0, 1.0)
        return torch.cat((self.pretrain_lambdas, task_lambdas), dim=1)

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def forward(self, input_ids, attention_mask, labels):
        lambdas = self.lambdas()
        params = [sum(pi * lambdasi for pi, lambdasi in zip(p, lambdas[0])) for p in zip(*self.paramslist)]
        params = [p.cuda() for p in params]
        load_weights(self.model, self.names, params)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

# Tokenizer and model initialization
tokenizer = T5Tokenizer.from_pretrained(args.model)
hf_t5_model = T5ForConditionalGeneration.from_pretrained(pretrained_checkpoint)
pretrained_model = SimpleCallableHFModel(hf_t5_model)
model = ModelWrapper(pretrained_model).to(args.device)
_, names = make_functional(model)

# Parameter list
paramslist = [tuple(v.detach().cpu().requires_grad_(False) for _, v in hf_t5_model.state_dict().items())]
paramslist += [tuple(v.detach().cpu().requires_grad_(False) for _, v in tv.vector.items()) for tv in task_vectors]

# Initialize AdaMerging
adamerging_mtl_model = AdaMerging(paramslist, model, names)

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
        outputs = adamerging_mtl_model(input_ids, attention_mask, labels)
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