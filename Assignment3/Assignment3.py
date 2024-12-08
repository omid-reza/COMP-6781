from datasets import load_dataset
import re
import string
from nltk.corpus import stopwords
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.amp import GradScaler, autocast

epoch_count_training = 10
dataset = load_dataset("IWSLT/iwslt2017",'iwslt2017-en-fr')
trimmed_dataset= dataset['train']['translation'][:100000]
nltk.download('stopwords')

def preprocess_data(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join([word for word in text.split(" ") if word.isalpha()])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def create_dataset(dataset, source_lang, target_lang):
    new_dataset = []
    for example in dataset:
        source_text = example.get(source_lang, "")
        target_text = example.get(target_lang, "")
        clean_source = preprocess_data(source_text)
        clean_target = preprocess_data(target_text)
        new_dataset.append((clean_source, clean_target))
    return new_dataset

training_set = create_dataset(trimmed_dataset, 'en', 'fr')
validation_set = create_dataset(dataset['validation']['translation'], 'en', 'fr')
test_set = create_dataset(dataset['test']['translation'], 'en', 'fr')

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)  # Embedding layer for source language
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)  # Embedding layer for target language
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensure the batch dimension is first
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)  # Last linear layer

    def positional_encoding(self, d_model, maxlen=5000):
        """Method to create a positional encoding buffer."""
        pos = torch.arange(0, maxlen).unsqueeze(1)
        denominator = 10000 ** (torch.arange(0, d_model, 2) / d_model)

        PE = torch.zeros((maxlen, d_model))
        PE[:, 0::2] = torch.sin(pos / denominator)  # Calculate sin for even positions
        PE[:, 1::2] = torch.cos(pos / denominator)  # Calculate cosine for odd positions

        PE = PE.unsqueeze(0)  # Add batch dimension

        return PE

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Method to forward a batch of data through the model."""
        # Pass source and target through embedding layer
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        positional_encoding = self.positional_encoding(d_model=src.shape[2]).to(src.device)  # Get positional encoding and move it to device

        # Get src_emb and tgt_emb by adding positional encoder
        src_emb = src + positional_encoding[:, :src.shape[1], :]
        tgt_emb = tgt + positional_encoding[:, :tgt.shape[1], :]

        # Pass src, tgt, and all masks through transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask, tgt_mask,
            None,  # Memory mask not used here
            src_key_padding_mask, tgt_key_padding_mask,
            src_key_padding_mask
        )

        # Pass output through linear layer
        output = self.fc(output)
        return output

    def encode(self, src, src_mask):
        """Method to encode a batch of data through the transformer model."""
        src = self.src_embedding(src)  # Pass src through embedding layer
        positional_encoding = self.positional_encoding(d_model=src.shape[2]).to(src.device)  # Create positional encoding
        src_emb = src + positional_encoding[:, :src.shape[1], :]  # Get src_emb
        return self.transformer.encoder(src_emb, src_mask)  # Pass src_emb through transformer encoder

    def decode(self, tgt, memory, tgt_mask):
        """Method to decode a batch of data through the transformer model."""
        tgt = self.tgt_embedding(tgt)  # Pass tgt through embedding layer
        positional_encoding = self.positional_encoding(d_model=tgt.shape[2]).to(tgt.device)  # Create positional encoding
        tgt_emb = tgt + positional_encoding[:, :tgt.shape[1], :]  # Get tgt_emb
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)  # Pass tgt_emb through transformer decoder


import torch

def create_padding_mask(seq):
    return (seq == 0).float()

def create_triu_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    mask = torch.flip(mask, dims=(0, 1))
    print(mask)
    return mask

def tokenize_batch(source, targets, tokenizer):
    tokenized_source = tokenizer(source, padding='max_length', max_length=120, truncation=True, return_tensors='pt')
    tokenized_targets = tokenizer(targets, padding='max_length', max_length=120, truncation=True, return_tensors='pt')
    return tokenized_source['input_ids'], tokenized_targets['input_ids']



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
PAD_IDX = tokenizer.pad_token_id #for padding
BOS_IDX = tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.pad_token_id #for beggining of sentence
EOS_IDX = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.pad_token_id #for end of sentence

model = TransformerModel(tokenizer.vocab_size, tokenizer.vocab_size,512, 8, 3, 3, 256,0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)

from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
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

        # Mixed precision forward pass
        with autocast(device_type='cuda'):  # Specify the device type
            logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss = loss / accumulation_steps

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Perform optimizer step after accumulation_steps
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
            src, tgt = tokenize_batch(src, tgt, tokenizer)
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            src_mask = torch.zeros((src.size(1), src.size(1)), device=device)  # Sequence x Sequence mask filled with zeros
            tgt_mask = create_triu_mask(tgt_input.size(1)).to(device)  # Create triangular mask for target
            src_padding_mask = create_padding_mask(src).to(device)  # Create padding mask for source
            tgt_padding_mask = create_padding_mask(tgt_input).to(device)  # Create padding mask for target
            logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )
            tgt_out = tgt[:, 1:]  # Shifted target output
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()  # Accumulate loss
    return losses / len(list(val_dataloader))



def train(model, epochs, train_loader, validation_loader):
    scaler = GradScaler()  # Initialize the gradient scaler
    for epoch in range(1, epochs + 1):
        # Pass the scaler to train_epoch
        train_loss = train_epoch(model, train_loader, tokenizer, scaler)
        val_loss = evaluate(model, validation_loader)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

train(model, epoch_count_training, train_loader, validation_loader)


import torch
from collections import defaultdict
from evaluate import load
bertscore = load("bertscore")
rouge = load('rouge')
meteor = load('meteor')


def greedy_decode(model, src, src_mask, max_len, start_symbol, repetition_penalty=1.5, top_k=10, max_repetitions=5):
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Pass through encoder
    memory = model.encode(src, src_mask)

    # Start decoding with the <BOS> token
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    repetition_counter = defaultdict(int)

    for i in range(max_len - 1):
        # Generate target mask
        tgt_mask = create_triu_mask(ys.size(1)).to(device)

        # Pass through the decoder to get logits
        out = model.decode(ys, memory, tgt_mask)
        logits = model.fc(out[:, -1])

        # Apply repetition penalty
        for token_id, count in repetition_counter.items():
            if count > 0:
                logits[0, token_id] /= (repetition_penalty ** count)

        # Apply top-k sampling
        topk_prob, topk_indices = torch.topk(logits, top_k, dim=-1)
        next_word_index = torch.multinomial(torch.nn.functional.softmax(topk_prob, dim=-1), 1).item()
        next_word = topk_indices[0, next_word_index].item()

        # Stop if end-of-sequence token is generated
        if next_word == EOS_IDX:
            break

        # Update repetition counter
        repetition_counter[next_word] += 1

        # Append the generated token to the target sequence (ys)
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        # Early stopping for excessive repetition
        if repetition_counter[next_word] >= max_repetitions:
            break

    return ys

def translate(model: torch.nn.Module, src_sentence: str, tokenizer):
    model.eval()
    src, _ = tokenize_batch(src_sentence, "", tokenizer)
    src = src.to(device)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.float).to(device)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len= int(num_tokens * 1.2 ), start_symbol=BOS_IDX).flatten()
    return tokenizer.decode(tgt_tokens, skip_special_tokens=True)

print(translate(model, "Hello how are you today", tokenizer))
print(f"BOS_IDX: {BOS_IDX}, EOS_IDX: {EOS_IDX}")
print(translate(model, "Hi", tokenizer))


import numpy as np
from evaluate import load

# Load evaluation metrics
bertscore = load("bertscore")
meteor = load('meteor')

def test(test_loader, model, tokenizer, device, max_length=200):
    precision = 0
    recall = 0
    f1 = 0
    meteor_metric = 0
    for src, target in test_loader:
        src_tensor, _ = tokenize_batch([src], [""], tokenizer)
        src_tensor = src_tensor.to(device)
        translated_output = translate(model, tokenizer.decode(src_tensor[0]), tokenizer)
        target_sentence = target[0]
        bert_results = bertscore.compute(predictions=[translated_output], references=[target_sentence], lang='fr')
        meteor_results = meteor.compute(predictions=[translated_output], references=[target_sentence])
        precision += bert_results['precision'][0]
        recall += bert_results['recall'][0]
        f1 += bert_results['f1'][0]
        meteor_metric += meteor_results['meteor']
    num_samples = len(test_loader)
    return precision / num_samples, recall / num_samples, f1 / num_samples, meteor_metric / num_samples

precision, recall, f1, meteor_metric = test(test_set, model, tokenizer, device)
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, METEOR: {meteor_metric:.3f}")

for i in range(3):
    src_sentence, _ = test_set[i]  #
    print(f"Source: {src_sentence}")
    translated = translate(model, src_sentence, tokenizer)
    print(f"Translation: {translated}")
    print('-' * 50)


import os
os.environ["WANDB_DISABLED"] = "true"


# Download nltk data
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_t5_input_for_training(dataset, source_lang="English", target_lang="French"):
    inputs = [f"translate {source_lang} to {target_lang}: {src}" for src, tgt in dataset]
    targets = [tgt for src, tgt in dataset]
    return inputs, targets

train_inputs, train_targets = preprocess_t5_input_for_training(training_set)
val_inputs, val_targets = preprocess_t5_input_for_training(validation_set)
test_inputs, test_targets = preprocess_t5_input_for_training(test_set)


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the T5 model and tokenizer
model_name = "t5-small"  # or "t5-base" for a larger model
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

print("T5 model and tokenizer loaded successfully!")

def translate_with_t5(t5_model, t5_tokenizer, sentences, device, max_length=200):
    inputs = t5_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    outputs = t5_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length)
    translations = t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations

def evaluate_translation_model(model, tokenizer, test_sentences, reference_sentences, device, is_t5=False, max_length=200):
    generated_translations = []
    meteor_metric = 0

    # Translate sentences
    for src_sentence in test_sentences:
        if is_t5:
            inputs = tokenizer(src_sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            translation = translate(model, src_sentence, tokenizer)
        generated_translations.append(translation)

    # Compute BERTScore
    P, R, F1 = bert_score(generated_translations, reference_sentences, lang="en")
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()

    # Compute METEOR score
    for ref, hyp in zip(reference_sentences, generated_translations):
        meteor_metric += single_meteor_score(ref.split(), hyp.split())

    return generated_translations, precision, recall, f1, meteor_metric / len(reference_sentences)

test_src_sentences = [src for src, _ in test_set[:10]]
test_ref_sentences = [tgt for _, tgt in test_set[:10]]

generated_translations, precision, recall, f1, meteor_metric = evaluate_translation_model(
    model=t5_model,
    tokenizer=t5_tokenizer,
    test_sentences=test_src_sentences,
    reference_sentences=test_ref_sentences,
    device=device,
    is_t5=True
)

print("Sample Translations:")
for i in range(5):  # Display first 5 translations
    print(f"Source: {test_src_sentences[i]}")
    print(f"Generated Translation: {generated_translations[i]}")
    print(f"Reference Translation: {test_ref_sentences[i]}\n")

print(f"T5 Model - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, METEOR: {meteor_metric:.4f}")


from bert_score import score as bert_score  # For BERTScore
from evaluate import load as load_metric  # For loading ROUGE
from nltk.translate.meteor_score import single_meteor_score

rouge = load_metric('rouge')

def evaluate_translation_model(model, tokenizer, test_sentences, reference_sentences, device, is_t5=False, max_length=200):
    generated_translations = []
    meteor_metric = 0

    for src_sentence in test_sentences:
        if is_t5:
            inputs = tokenizer(src_sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            translation = translate(model, src_sentence, tokenizer)
        generated_translations.append(translation)

    P, R, F1 = bert_score(generated_translations, reference_sentences, lang="en")
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()
    rouge_scores = rouge.compute(predictions=generated_translations, references=reference_sentences)
    for ref, hyp in zip(reference_sentences, generated_translations):
        meteor_metric += single_meteor_score(ref.split(), hyp.split())

    return generated_translations, precision, recall, f1, meteor_metric / len(reference_sentences), rouge_scores

test_src_sentences = [src for src, _ in test_set[:10]]
test_ref_sentences = [tgt for _, tgt in test_set[:10]]

generated_translations, precision, recall, f1, meteor_metric, rouge_scores = evaluate_translation_model(
    model=t5_model,
    tokenizer=t5_tokenizer,
    test_sentences=test_src_sentences,
    reference_sentences=test_ref_sentences,
    device=device,
    is_t5=True
)

print("Sample Translations:")
for i in range(5):  # Display first 5 translations
    print(f"Source: {test_src_sentences[i]}")
    print(f"Generated Translation: {generated_translations[i]}")
    print(f"Reference Translation: {test_ref_sentences[i]}\n")

print(f"T5 Model - Precision (BERTScore): {precision:.4f}, Recall (BERTScore): {recall:.4f}, F1 (BERTScore): {f1:.4f}, METEOR: {meteor_metric:.4f}")
print(f"ROUGE Scores: {rouge_scores}")

avg_f1_score = (rouge_scores['rouge1'] +
                rouge_scores['rouge2'] +
                rouge_scores['rougeL']) / 3

print(f"Average ROUGE F1 Score: {avg_f1_score:.4f}")

print(rouge_scores)

avg_f1_score = (rouge_scores['rouge1'] +
                rouge_scores['rouge2'] +
                rouge_scores['rougeL'] +
                rouge_scores['rougeLsum']) / 4

print(f"Average ROUGE F1 Score (including ROUGE-Lsum): {avg_f1_score:.4f}")
