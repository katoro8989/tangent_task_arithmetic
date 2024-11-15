# download civil comments
# python prepare_civil_comments.py

import os
import json
import argparse
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


def create_and_save_comments(data_path):
    # Load the Civil Comments dataset
    civil_comments = load_dataset("google/civil_comments", split="train")

    # Filter comments with toxicity score greater than 0.8
    toxic_comments = civil_comments.filter(lambda x: x["toxicity"] > 0.8)

    # Initialize the GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Tokenize the filtered comments
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_comments = toxic_comments.map(tokenize_function, batched=True)

    # Save the tokenized dataset to disk
    tokenized_comments.save_to_disk(data_path)
    print(f"Tokenized dataset saved to {data_path}")

data_path = "/mnt2/dataset/civil_comments"
os.makedirs(data_path, exist_ok=True)

create_and_save_comments(data_path)