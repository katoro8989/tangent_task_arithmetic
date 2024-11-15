# download civil comments
# python prepare_civil_comments.py

import os
import json
import argparse
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer




#cratete comments where the toxicity score is higher than 0.8
#load google/civil_comments from huggingface
def create_comments(data_path, output_path):
    civil_comments = load_dataset("google/civil_comments")
    civil_comments = civil_comments["train"]
    civil_comments = civil_comments.filter(lambda x: x["toxicity"] > 0.8)
    #encode with gpt-2 tokenizer
    civil_comments = civil_comments.map(lambda x: gpt2_tokenizer(x["text"]), batched=True)
    with open(output_path, "w") as f:
        json.dump(civil_comments, f)



data_path = "/mnt2/dataset/civil_comments"
os.makedirs(data_path, exist_ok=True)
output_path = os.path.join(data_path, "comments.json")