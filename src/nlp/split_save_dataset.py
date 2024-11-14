from datasets import load_dataset, DatasetDict, load_from_disk
from dataset_preprocess.glue_process import get_preprocessor, get_map_kwargs
from torch.utils.data import DataLoader, Subset
import random
import os

TASKS = ["cola", "mrpc", "rte", "sst2"]
for task in TASKS:
    dataset_class = load_dataset("glue", task)
        
    preprocessor_class = get_preprocessor(task)
    preprocessor = preprocessor_class(tokenizer=tokenizer, tokenizer_kwargs=args.tokenizer_kwargs)
    map_kwargs = get_map_kwargs(task)
    encoded_dataset = dataset_class.map(preprocessor, **map_kwargs)

    num_validation_samples = len(encoded_dataset["validation"])

    train_indices = list(range(len(encoded_dataset["train"])))
    random.shuffle(train_indices)  # インデックスをシャッフル
    validation_indices = train_indices[:num_validation_samples]
    train_indices = train_indices[num_validation_samples:]

    new_train_dataset = Subset(encoded_dataset["train"], train_indices)
    new_validation_dataset = Subset(encoded_dataset["train"], validation_indices)
    test_dataset = encoded_dataset["validation"]

    output_dir = "/mnt2/dataset/glue_split"
    os.makedirs(output_dir, exist_ok=True)

    split_datasets = DatasetDict({
        "train": encoded_dataset["train"].select(train_indices),
        "validation": encoded_dataset["train"].select(validation_indices),
        "test": encoded_dataset["validation"]
    })

    split_datasets.save_to_disk(output_dir)
    print(f"Dataset {task} saved to {output_dir}")