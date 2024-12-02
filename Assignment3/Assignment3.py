from datasets import load_dataset
import string
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import numpy as np

dataset = load_dataset("IWSLT/iwslt2017",'iwslt2017-en-fr', trust_remote_code=True)
trim_dataset= dataset['train']['translation'][:100000]


def preprocess_data(text):
  text = text.lower() #make everything lower case
  text = text.replace("\n", " ") #remove \n characters
  text= re.sub(f"[{re.escape(string.punctuation)}]", "", text)#remove any punctuation or special characters
  text = re.sub(r"\d+", "", text)#remove all numbers
  return text

def create_dataset(dataset,source_lang,target_lang):
  new_dataset=[]
  for item in dataset:
    source_text = preprocess_data(item[source_lang])  # Preprocess source language
    target_text = preprocess_data(item[target_lang])  # Preprocess target language
    new_dataset.append((source_text, target_text))
  return new_dataset

training_set=create_dataset(trim_dataset,'en','fr')
validation_set=create_dataset(dataset['validation']['translation'],'en','fr')
test_set=create_dataset(dataset['test']['translation'],'en','fr')

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def positional_encoding(self, d_model, maxlen=5000):
        pos = torch.arange(0, maxlen).unsqueeze(1)
        denominator = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        PE = torch.zeros((maxlen, d_model))
        PE[:, 0::2] = torch.sin(pos / denominator)
        PE[:, 1::2] = torch.cos(pos / denominator)
        PE = PE.unsqueeze(0)
        return PE

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        positional_encoding = self.positional_encoding(d_model=src.shape[2]).to(src.device)  # Get positional encoding and move it to device
        src_emb = src + positional_encoding[:, :src.shape[1], :]
        tgt_emb = tgt + positional_encoding[:, :tgt.shape[1], :]
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask, tgt_mask,
            None,  # Memory mask not used here
            src_key_padding_mask, tgt_key_padding_mask,
            src_key_padding_mask
        )
        output = self.fc(output)
        return output

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        positional_encoding = self.positional_encoding(d_model=src.shape[2]).to(src.device)
        src_emb = src + positional_encoding[:, :src.shape[1], :]
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        positional_encoding = self.positional_encoding(d_model=tgt.shape[2]).to(tgt.device)
        tgt_emb = tgt + positional_encoding[:, :tgt.shape[1], :]
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)


def create_padding_mask(seq):
    return (seq == 0).float()

def create_triu_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).transpose(0, 1).float()
    return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

def tokenize_batch(source, targets, tokenizer):
    tokenized_source = tokenizer(source, padding='max_length', max_length=120, return_tensors='pt')
    tokenized_targets = tokenizer(targets, padding='max_length', max_length=120, return_tensors='pt')

    return tokenized_source['input_ids'], tokenized_targets['input_ids']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.pad_token_id
EOS_IDX = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.pad_token_id
model = TransformerModel(tokenizer.vocab_size, tokenizer.vocab_size,512, 8, 3, 3, 256,0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)


def train_epoch(model, train_loader, tokenizer, scaler, accumulation_steps=4):
    model.train()
    losses = 0
    optimizer.zero_grad()
    for batch_idx, (src, tgt) in enumerate(tqdm(train_loader)):
        src, tgt = tokenize_batch(src, tgt, tokenizer)
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = torch.zeros((src.size(1), src.size(1)), device=device)
        tgt_mask = create_triu_mask(tgt_input.size(1)).to(device)
        src_padding_mask = create_padding_mask(src).to(device)
        tgt_padding_mask = create_padding_mask(tgt_input).to(device)
        with autocast(device_type='cuda'):
            logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        losses += loss.item()
    return losses / len(list(train_loader))

def evaluate(model, val_dataloader):
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader):
            # Tokenize and move to device
            src, tgt = tokenize_batch(src, tgt, tokenizer)
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            src_mask = torch.zeros((src.size(1), src.size(1)), device=device)
            tgt_mask = create_triu_mask(tgt_input.size(1)).to(device)
            src_padding_mask = create_padding_mask(src).to(device)
            tgt_padding_mask = create_padding_mask(tgt_input).to(device)
            logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )
            tgt_out = tgt[:, 1:]
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    return losses / len(list(val_dataloader))

def train(model, epochs, train_loader, validation_loader):
    scaler = GradScaler()  # Initialize the gradient scaler
    for epoch in range(1, epochs + 1):
        # Pass the scaler to train_epoch
        train_loss = train_epoch(model, train_loader, tokenizer, scaler)
        val_loss = evaluate(model, validation_loader)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

train(model, 20, train_loader, validation_loader)


from evaluate import load
bertscore = load("bertscore")
rouge = load('rouge')
meteor = load('meteor')

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        tgt_mask = create_triu_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, tgt_mask)

        prob = model.fc(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str, tokenizer):
    model.eval()
    src, _ = tokenize_batch(src_sentence, "", tokenizer)
    src = src.to(device)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.float).to(device)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len= int(num_tokens * 1.2 ), start_symbol=tokenizer.cls_token_id).flatten()
    return tokenizer.decode(tgt_tokens, skip_special_tokens=True)

print(translate(model, "Hello how are you today", tokenizer))

def test(test_loader, model, tokenizer, device, max_length=200):
  precision = 0
  recall = 0
  f1 = 0
  meteor_metric = 0

  model.eval()
  for src, target in test_loader:
    src = [preprocess_data(s) for s in src]
    target = [preprocess_data(t) for t in target]
    predictions = []
    for sentence in src:
        translated_sentence = translate(model, sentence, tokenizer)
        predictions.append(translated_sentence)
    results_bert = bertscore.compute(predictions=predictions, references=target, lang='en')
    results_meteor = meteor.compute(predictions=predictions, references=target)
    precision += np.mean(results_bert['precision'])
    recall += np.mean(results_bert['recall'])
    f1 += np.mean(results_bert['f1'])
    meteor_metric+= results_meteor['meteor']
  return precision / len(test_loader), recall / len(test_loader), f1 / len(test_loader), meteor_metric / len(test_loader)

test(test_set, model, tokenizer, device)