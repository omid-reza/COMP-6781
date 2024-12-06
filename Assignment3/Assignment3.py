from datasets import load_dataset
import re
import string
from nltk.corpus import stopwords
import nltk


dataset = load_dataset("IWSLT/iwslt2017",'iwslt2017-en-fr')
trimmed_dataset= dataset['train']['translation'][:100000]
nltk.download('stopwords')

def preprocess_data(text):
    """ Method to clean text from noise and standardize text across the different classes.
        The preprocessing includes converting to lowercase, removing punctuation, and removing stopwords.
    Arguments
    ---------
    text : String
        Text to clean
    Returns
    -------
    text : String
        Cleaned text
    """

    # Make everything lower case
    text = text.lower()

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Remove any punctuation or special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove all numbers
    text = ' '.join([word for word in text.split(" ") if word.isalpha()])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def create_dataset(dataset, source_lang, target_lang):
    """
    Method to create a dataset from a list of text.

    Arguments
    ---------
    dataset : List of Dict
        List of dictionary objects with source and target text
    source_lang : String
        Source language key in the dataset
    target_lang : String
        Target language key in the dataset

    Returns
    -------
    new_dataset : List of Tuples
        Cleaned source and target text in format (source, target)
    """
    new_dataset = []
    for example in dataset:
        # Extract source and target text
        source_text = example.get(source_lang, "")
        target_text = example.get(target_lang, "")

        # Preprocess source and target text
        clean_source = preprocess_data(source_text)
        clean_target = preprocess_data(target_text)

        # Append to the dataset
        new_dataset.append((clean_source, clean_target))
    return new_dataset

# Applying the preprocessing and formatting the training, validation, and test sets
training_set = create_dataset(trimmed_dataset, 'en', 'fr')
validation_set = create_dataset(dataset['validation']['translation'], 'en', 'fr')
test_set = create_dataset(dataset['test']['translation'], 'en', 'fr')


import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """
    Method to create a padding mask based on given sequence.
    Arguments
    ---------
    seq : Tensor
       Sequence to create padding mask for
    Returns
    -------
    mask : Tensor
        Padding mask
    """
    # Create a mask where padded tokens (value 0) are marked as 1
    mask = (seq == 0).float()
    return mask  # Return a 2-D tensor with shape (batch_size, sequence_length)

def create_triu_mask(sz):
    """
    Method to create a triangular mask based on given sequence.
    This is used for the tgt mask in the Transformer model to avoid looking ahead.
    Arguments
    ---------
    sz : int
       Size of the mask
    Returns
    -------
    mask : Tensor
        Triangular mask
    """
    # Create a lower triangular matrix with 1's in the lower triangle and 0's elsewhere
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).transpose(0, 1).float()
    # Replace 0 with 0.0 and 1 with -inf
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

def tokenize_batch(source, targets, tokenizer):
    """
    Method to tokenize a batch of data given a tokenizer.
    Arguments
    ---------
    source : List of String
       Source text
    targets : List of String
       Target text
    tokenizer : Tokenizer
       Tokenizer to use for tokenization
    Returns
    -------
    tokenized_source : Tensor
        Tokenized source text
    tokenized_targets : Tensor
        Tokenized target text
    """
    tokenized_source = tokenizer(source, padding='max_length', max_length=120, truncation=True, return_tensors='pt')
    tokenized_targets = tokenizer(targets, padding='max_length', max_length=120, truncation=True, return_tensors='pt')


    return tokenized_source['input_ids'], tokenized_targets['input_ids']


from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
# PAD_IDX = tokenizer.pad_token_id #for padding
# BOS_IDX = tokenizer.bos_token_id #for beggining of sentence
# EOS_IDX = tokenizer.eos_token_id #for end of sentence

PAD_IDX = tokenizer.pad_token_id #for padding
BOS_IDX = tokenizer.cls_token_id if tokenizer.cls_token_id else tokenizer.pad_token_id #for beggining of sentence
EOS_IDX = tokenizer.sep_token_id if tokenizer.sep_token_id else tokenizer.pad_token_id #for end of sentence

model = TransformerModel(tokenizer.vocab_size, tokenizer.vocab_size,512, 8, 3, 3, 256,0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# Update DataLoader batch size
train_loader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=False)

from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
def train_epoch(model, train_loader, tokenizer, scaler, accumulation_steps=4):
    model.train()
    losses = 0
    optimizer.zero_grad()  # Initialize gradients

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
            # Tokenize and move to device
            src, tgt = tokenize_batch(src, tgt, tokenizer)
            src = src.to(device)
            tgt = tgt.to(device)

            # Shift target for teacher forcing
            tgt_input = tgt[:, :-1]

            # Create masks
            src_mask = torch.zeros((src.size(1), src.size(1)), device=device)  # Sequence x Sequence mask filled with zeros
            tgt_mask = create_triu_mask(tgt_input.size(1)).to(device)  # Create triangular mask for target

            src_padding_mask = create_padding_mask(src).to(device)  # Create padding mask for source
            tgt_padding_mask = create_padding_mask(tgt_input).to(device)  # Create padding mask for target

            # Forward pass through the model
            logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )

            # Compute loss
            tgt_out = tgt[:, 1:]  # Shifted target output
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()  # Accumulate loss

    return losses / len(list(val_dataloader))


from torch.amp import GradScaler, autocast

def train(model, epochs, train_loader, validation_loader):
    scaler = GradScaler()  # Initialize the gradient scaler
    for epoch in range(1, epochs + 1):
        # Pass the scaler to train_epoch
        train_loss = train_epoch(model, train_loader, tokenizer, scaler)
        val_loss = evaluate(model, validation_loader)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

train(model, 100, train_loader, validation_loader)


import torch
from collections import defaultdict
from evaluate import load
bertscore = load("bertscore")
rouge = load('rouge')
meteor = load('meteor')


def greedy_decode(model, src, src_mask, max_len, start_symbol, repetition_penalty=1.5, top_k=10, max_repetitions=5):
    """
    Greedy decoding with repetition penalty and top-k sampling without temperature.

    Args:
        model: The trained model.
        src: Input sequence tensor.
        src_mask: Mask for the input sequence.
        max_len: Maximum output length.
        start_symbol: The ID of the starting token.
        repetition_penalty: Penalize repeated tokens in logits (default: 1.5).
        top_k: The number of top tokens to consider in the sampling (default: 10).
        max_repetitions: Number of repetitive tokens to trigger early stop.

    Returns:
        Tensor of generated token IDs.
    """
    # Move source to device
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
print(translate(model, "I hope you are doing well", tokenizer))
