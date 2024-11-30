from datasets import load_dataset
dataset = load_dataset("IWSLT/iwslt2017",'iwslt2017-en-fr')
trim_dataset= dataset['train']['translation'][:100000]

import string
import re
def preprocess_data(text):
  """ Method to clean text from noise and standarize text across the different classes.
      The preprocessing includes converting to joining all datapoints, lowercase, removing punctuation, and removing stopwords.
  Arguments
  ---------
  text : List of String
     Text to clean
  Returns
  -------
  text : String
      Cleaned and joined text
  """

  text = text.lower() #make everything lower case
  text = text.replace("\n", " ") #remove \n characters
  text= re.sub(f"[{re.escape(string.punctuation)}]", "", text)#remove any punctuation or special characters
  text = re.sub(r"\d+", "", text)#remove all numbers

  return text

def create_dataset(dataset,source_lang,target_lang):
  """ Method to create a dataset from a list of text.
  Arguments
  ---------
  text : List of String
     Text from dataset
  source_lang : String
     Source language
  target_lang : String
     Target language
  Returns
  -------
  new_dataset : Tuple of String
      Source and target text in format (source, target)
  """
  new_dataset=[]
  #TODO: iterate through dataset extract source and target dataset and preprocess them creating a new clean dataset with the correct format
  for item in dataset:
    source_text = preprocess_data(item[source_lang])  # Preprocess source language
    target_text = preprocess_data(item[target_lang])  # Preprocess target language
    new_dataset.append((source_text, target_text))
  return new_dataset

training_set=create_dataset(trim_dataset,'en','fr')
validation_set=create_dataset(dataset['validation']['translation'],'en','fr')
test_set=create_dataset(dataset['test']['translation'],'en','fr')

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,dropout):
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model) # Embedding layer for source language
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model) # Embedding layer for target language
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        ) # Transformer model with it's attributes (see pytorch documentation), set batch_first to True
        self.fc = nn.Linear(d_model, tgt_vocab_size) # Last linear layer

    def positional_encoding(self, d_model, maxlen = 5000):
        """Method to create a positional encoding buffer.
        Arguments
        ---------
        d_model: int
            Embedding size
        maxlen: int
            Maximum sequence length
        Returns
        -------
        PE: Tensor
            Positional encoding buffer
        """
        pos = torch.arange(0, maxlen).unsqueeze(1)
        denominator = 10000 ** (torch.arange(0, d_model, 2) / d_model)

        PE = torch.zeros((maxlen, d_model))
        PE[:, 0::2] = torch.sin(pos / denominator) # Calculate sin for even positions
        PE[:, 1::2] = torch.cos(pos / denominator) # Calculate cosine for odd positions

        PE = PE.unsqueeze(0)  # Add batch dimension

        return PE


    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Method to forward a batch of data through the model."""
        #pass source and target throught embedding layer
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        positional_encoding = self.positional_encoding(src.size(2)).to(src.device) #get positional encoding and move it to device

        #get src_emb and tgt_emb by adding positional encoder
        src_emb = src + positional_encoding[:,:src.shape[1], :]
        tgt_emb = tgt + positional_encoding[:,:tgt.shape[1], :]

        #pass src, tgt and all masks throught transformer
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_key_padding_mask, tgt_key_padding_mask,src_key_padding_mask)

        #pass output throught linear layer
        output = self.fc(output)
        return output

    def encode(self, src, src_mask):
        """Method to encode a batch of data through the transformer model."""
        src = self.src_embedding(src) #pass src throught embedding layer
        positional_encoding = self.positional_encoding(src.size(2)).to(src.device) #create positional encoding
        src_emb = src + positional_encoding[:, :src.shape[1], :] #get src_emb
        return self.transformer.encoder(src_emb, src_mask) #pass src_emb through transformer encoder (look pytorch documentation)


    def decode(self, tgt, memory,tgt_mask):
        """Method to decode a batch of data through the transformer model."""
        tgt = self.tgt_embedding(tgt) #pass tgt throught embedding layer
        positional_encoding = self.positional_encoding(tgt.size(2)).to(tgt.device) #create positional encoding
        tgt_emb = tgt + positional_encoding[:, :tgt.shape[1], :] #get tgt_emb
        return self.transformer.decoder(tgt_emb, memory, tgt_mask) #pass tgt_emb through transformer decoder (look pytorch documentation)


def create_padding_mask(seq):
  """ Method to create a padding mask based on given sequence.
  Arguments
  ---------
  seq : Tensor
     Sequence to create padding mask for
  Returns
  -------
  mask : Tensor
      Padding mask
  """
  return (seq == PAD_IDX).float() #float matrix that is 1 when datapoint is equal to 0

def create_triu_mask(sz):
  """ Method to create a triangular mask based on given sequence. This is used for the tgt mask in the Transformer model to avoid looking ahead.
  Arguments
  ---------
  seq : Tensor
     Sequence to create triangular mask for
  Returns
  -------
  mask : Tensor
      Triangular mask
  """
  # TODO
  mask = torch.triu(torch.ones(sz, sz), diagonal=1) #create triangular mask of size sz x sz
  mask = mask.transpose(0, 1).float() #tranpose mask and cast to float type
  mask[mask == 1] = float('-inf') #in pytorch the masked objects expect -inf instead of zero. Replace all 0 for -inf and all 1's for 0's
  mask[mask == 0] = 0.0 #you might want to transpose at the end
  return mask

def tokenize_batch(source, targets,tokenizer):
  """ Method to tokenize a batch of data given a tokenizer.
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
  """

  tokenized_source = tokenizer(source, padding='max_length', max_length=120, return_tensors='pt')

  tokenized_targets = tokenizer(targets,  padding='max_length', max_length=120, return_tensors='pt')

  return tokenized_source['input_ids'], tokenized_targets['input_ids']


from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer=AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-uncased')
PAD_IDX = tokenizer.pad_token_id #for padding
BOS_IDX = tokenizer.bos_token_id #for beggining of sentence
EOS_IDX = tokenizer.eos_token_id #for end of sentence

model = TransformerModel(tokenizer.vocab_size, tokenizer.vocab_size,512, 8, 3, 3, 256,0.1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)



from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model,train_loader,tokenizer):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_loader):
        src, tgt = tokenize_batch(src, tgt, tokenizer)
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:,:-1]

        #TODO
        src_mask = torch.zeros((src.size(1), src.size(1)), device=device) #creat src_mask this is basically a matrix of 0s of shape Sequence x Sequence (see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
        tgt_mask = create_triu_mask(tgt_input.size(1)).to(device) #create triangular mask for target

        src_padding_mask = create_padding_mask(src).to(device) #create padding mask for src
        tgt_padding_mask = create_padding_mask(tgt_input).to(device) #create padding mask for tgt

        logits = model(
            src, tgt_input,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
        ) #pass it through model

        optimizer.zero_grad()

        tgt_out = tgt[:,1:]
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))


def evaluate(model,val_dataloader ):
    model.eval()
    losses = 0
    with torch.no_grad():
      for src, tgt in tqdm(val_dataloader):
          src, tgt = tokenize_batch(src, tgt, tokenizer)
          src = src.to(device)
          tgt = tgt.to(device)

          tgt_input = tgt[:,:-1]

          #do the same as in Train

          src_mask = torch.zeros((src.size(1), src.size(1)), device=device)  # Source mask
          tgt_mask = create_triu_mask(tgt_input.size(1)).to(device)  # Triangular mask for target
          src_padding_mask = create_padding_mask(src).to(device)  # Padding mask for source
          tgt_padding_mask = create_padding_mask(tgt_input).to(device)
          logits = model(
                src, tgt_input,
                src_mask=src_mask, tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask
            )


          tgt_out = tgt[:,1:]
          loss = loss_function(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
          losses += loss.item()

    return losses / len(list(val_dataloader))

def train(model, epochs, train_loader,validation_loader ):
  for epoch in range(1, epochs+1):
        train_loss = train_epoch(model,train_loader, tokenizer)
        val_loss = evaluate(model,validation_loader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"))

train(model, 10, train_loader,validation_loader)