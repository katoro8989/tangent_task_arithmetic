
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import sys
from tqdm import tqdm
import torch
import gc
from src.task_vectors import NonLinearTaskVector
from src.eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from src.args import parse_arguments
import wandb

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

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-32'
args = parse_arguments()
args.model = model

run = wandb.init(config=vars(args), 
                 project=f"{args.model}_ada",
                 entity='katoro13',
                 )

if args.seed is not None:
    args.save = f"/mnt/data/checkpoints{args.seed}/{args.model}"
else:
    args.save = f"/mnt/data/checkpoints/{args.model}"

args.logs_path = args.save
pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_Task_wise_AdaMerging.txt'.format(str_time_))
args.log = log

task_vectors = [NonLinearTaskVector(pretrained_checkpoint, f"{args.save}/{dataset_name}Val/finetuned.pt") for dataset_name in exam_datasets]

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

    def forward(self, images):
        features = self.model(images)
        return features

from heads import get_classification_head
class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

pretrained_model = torch.load(pretrained_checkpoint)
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

from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

dataloaders = {dataset_name: get_dataloader_shuffle(get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=16)) for dataset_name in exam_datasets}
data_iters = {dataset_name: iter(dataloaders[dataset_name]) for dataset_name in exam_datasets}

for epoch in range(epochs):
    start_time = time.time()
    losses = 0.
    for dataset_name in exam_datasets:
        dataloader = dataloaders[dataset_name]

        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            outputs = adamerging_mtl_model(x, dataset_name)
            loss = softmax_entropy(outputs).mean(0)
            losses += loss

            if i > 0:  
                break

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    end_time = time.time()
    print("TIME", end_time - start_time)

    print(list(adamerging_mtl_model.lambdas().data))

    if ((epoch+1) % 500) == 0:
        log.info(str(list(adamerging_mtl_model.lambdas().data)))

        Total_ACC = 0.
        for dataset_name in exam_datasets:
            image_encoder = adamerging_mtl_model.get_image_encoder()
            classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
            metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
            Total_ACC += metrics['top1']
            log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
        log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(exam_datasets)) + '\n')
