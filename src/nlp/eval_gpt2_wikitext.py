import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import numpy as np

parser = argparse.ArgumentParser(description='Finetuning of T5')
parser.add_argument('--finetuning_mode', type=str, default="standard")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device_number', type=int, default=0)
args = parser.parse_args()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

model = GPT2LMHeadModel.from_pretrained(args.model)
model.resize_token_embeddings(len(tokenizer))
model = SimpleCallableHFModel(model)

if args.finetuning_mode == "linear":
    linearized_finetuning = True
    model = LinearizedModelWrapper(model)
else:
    linearized_finetuning = False

model = model.to(device)

# WikiText-103データセットのロード
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

# パープレキシティ計算用の関数
def calculate_perplexity(model, tokenizer, dataset, stride=512, max_length=1024):
    model.eval()
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for sample in dataset:
            input_text = sample["text"]
            encodings = tokenizer(input_text, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            
            # スライディングウィンドウでパープレキシティを計算
            for i in range(0, input_ids.size(1), stride):
                input_slice = input_ids[:, i:i + max_length]
                if input_slice.size(1) < max_length:
                    break  # ウィンドウの長さが指定サイズより小さい場合スキップ
                
                labels = input_slice.clone()
                outputs = model(input_slice, labels=labels)
                loss = outputs.loss
                batch_loss = loss.item() * input_slice.size(1)  # スライスのトークン数で重み付け
                
                total_loss += batch_loss
                total_length += input_slice.size(1)
    
    # パープレキシティの計算
    avg_loss = total_loss / total_length
    perplexity = np.exp(avg_loss)
    return perplexity

# パープレキシティを計算
perplexity = calculate_perplexity(model, tokenizer, dataset, stride=512, max_length=1024)
print(f"Perplexity: {perplexity}")