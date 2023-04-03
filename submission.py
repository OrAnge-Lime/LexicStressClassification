import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.functional import pad
from Git_repo.LexicStressClassification.SoundClassifierTransformer import LexicStressClassification
from Git_repo.LexicStressClassification.SoundClassifierTransformer import TokenEmbedding
from Git_repo.LexicStressClassification.SoundClassifierTransformer import PositionalEncoding

import pandas as pd

vocab = torch.load("results/vocab_large.pt")
MAX_LEN = 6

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

tokenizer = get_tokenizer(token_function)

def predict(word):
    with torch.no_grad():
        tokenized_word = vocab(tokenizer(word))
        tokenized_word = torch.tensor(tokenized_word, dtype=torch.int64)
        paded_tensor = pad(tokenized_word, (0, MAX_LEN-len(tokenized_word)))

        paded_tensor = torch.unsqueeze(paded_tensor, 0)

        padding_mask = (paded_tensor == 0)
        target_embed = model.pos_encode(model.token_embedding(paded_tensor))
        output = model.transformer(target_embed, None, padding_mask)

        output = torch.flatten(output)
        output = model.out_layer(output)
    
        return output
    
model = torch.load("results/final_model_large.pt")
model.to("cpu")
model.eval()

df = pd.read_csv("data/test.csv")

with open("results/submission_largev2.csv", "x") as f:
    for idx, w in df.iterrows():
        stress = torch.argmax(predict(w["word"])).item() + 1
        id = w["id"]
        f.write(f"{id},{stress}\n")