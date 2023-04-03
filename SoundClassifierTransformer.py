import torch
from torch.utils.data import DataLoader
from torch import nn
import time
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.functional import pad
from tqdm import tqdm
from torch.utils.data import random_split
import math
import os

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  
print(device)

DATA_PATH = "data"
RES_PATH = "results"

FILE_PATH = os.path.join(DATA_PATH, "train.csv")
df = pd.read_csv(FILE_PATH) 

VOWELS = "аеёиоыуэюя"

def token_function(word):
    res = []
    syllable = ""
    for letter in word:
        syllable += letter
        if letter in VOWELS:
            res.append(syllable)
            syllable = ""

    if not any([x in VOWELS for x in syllable]):
        res[-1] += syllable
    else:
        res.append(syllable)

    return res

PAD = "<pad>"
UNKNOWN = "<unk>"

tokenizer = get_tokenizer(token_function)

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(df['word']), specials=[PAD, UNKNOWN])
vocab.set_default_index(vocab[UNKNOWN])
torch.save(vocab, os.path.join(RES_PATH, "vocab_large.pt"))

class LexicStressDataset(Dataset):
  def __init__(self, text, labels):
        self.labels = labels
        self.text = text

  def __len__(self):
        return len(self.text)

  def __getitem__(self, index):
      word = self.text[index]
      label = self.labels[index]
      return label, word

dataset = LexicStressDataset(df['word'], df['stress'])

MAX_LEN = max([sum([i in VOWELS for i in x]) for x in df["word"]]) 

def collate_batch(batch):
    label_list, word_list = [], torch.IntTensor()

    for (label, word) in batch:
        label_list.append(int(label)-1)

        tokenized_word = vocab(tokenizer(word))
        tokenized_word = torch.tensor(tokenized_word, dtype=torch.int64)
        paded_tensor = pad(tokenized_word, (0, MAX_LEN-len(tokenized_word)))
        
        if len(word_list):
            word_list = torch.cat((word_list, paded_tensor))
        else:
            word_list = paded_tensor
            
    label_list = torch.tensor(label_list, dtype=torch.int64)
    word_list = word_list.reshape(-1, MAX_LEN)
    
    return label_list.to(device), word_list.to(device)

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class LexicStressClassification(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(LexicStressClassification, self).__init__()
        self.dropout = nn.Dropout()

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, dropout=0.1)

        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    batch_first=True,
            ), num_layers=2)

        self.out_layer = nn.LazyLinear(num_class)

    def forward(self, word):
        padding_mask = (word == 0)
        target_embed =self.pos_encode(self.token_embedding(word))
        output = self.transformer(target_embed, None, padding_mask)

        output = self.dropout(output)
        output = torch.flatten(output, start_dim=1)
        output = self.out_layer(output)

        return output

model = LexicStressClassification(len(vocab), 256, 6).to(device)

EPOCHS = 40
LR = 0.0005
BATCH_SIZE = 128

train_data, val_data = random_split(dataset, [round(len(dataset)*0.8), len(dataset)-round(len(dataset)*0.8)])

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# train_full_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def accuracy(y_pred, y_true):
    a = torch.sum(torch.argmax(y_pred, dim=-1) == y_true)
    return a / y_pred.shape[0]

def train(dataloader):
    model.train() 

    for idx, (label, word) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad() 
        prediction = model(word)
        loss = loss_function(prediction, label)
        loss.backward()

        optimizer.step()

        if idx % 500 == 0:
            print("loss: ", loss.item(), "accuracy: ", accuracy(prediction, label).item())

def eval(dataloader):
    model.eval()
    total_accuracy = 0
    
    for _, (label, word) in enumerate(dataloader):
        prediction = model(word)
        total_accuracy += accuracy(prediction, label)

    return total_accuracy/len(dataloader)

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        val_acc = None
        val_acc = eval(val_dataloader)
        end_time = time.time() - epoch_start_time

        print('-' * 59)
        print(f'| end of epoch {epoch} | time: {end_time}s | validation accuracy: {val_acc}')
        print('-' * 59)

    torch.save(model, os.path.join(RES_PATH, "final_model_large.pt"))