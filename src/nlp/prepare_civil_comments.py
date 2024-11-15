import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def create_and_save_comments(data_path):
    # Load the Civil Comments dataset (train and validation)
    civil_comments = load_dataset("google/civil_comments")

    # Filter comments with toxicity score greater than 0.8 for both train and validation
    toxic_train = civil_comments["train"].filter(lambda x: x["toxicity"] > 0.8)
    toxic_validation = civil_comments["validation"].filter(lambda x: x["toxicity"] > 0.8)

    # Display the sizes of the filtered datasets
    print(f"Number of samples in filtered train set: {len(toxic_train)}")
    print(f"Number of samples in filtered validation set: {len(toxic_validation)}")

    # Initialize the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Tokenize the filtered comments
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = toxic_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_validation = toxic_validation.map(tokenize_function, batched=True, remove_columns=["text"])

    # Combine into a DatasetDict
    tokenized_datasets = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_validation
    })

    # Save the tokenized DatasetDict to disk
    tokenized_datasets.save_to_disk(data_path)
    print(f"Tokenized dataset saved to {data_path}")

# Set the data path
data_path = "/mnt2/dataset/civil_comments"
os.makedirs(data_path, exist_ok=True)

# Create and save the tokenized dataset
create_and_save_comments(data_path)