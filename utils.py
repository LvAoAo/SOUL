import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchtext.vocab import FastText
import nltk
# from nltk.corpus import brown, gutenberg, webtext, nps_chat, reuters, inaugural
import joblib

def tokenlize(path="data", pred_word=None):
    pro_path = path + "/processed.xlsx"
    df = pd.read_excel(pro_path)
    # print(list(df["Word"]))
    tokens = list(df["Word"])
    try_matrix = torch.tensor(np.array(df[["1 try","2 tries","3 tries","4 tries","5 tries","6 tries",
                                           "7 or more tries (X)"]]), dtype=torch.float)
    embedding_dim = 300
    text_vocab = FastText()
    wrong_indices, indices_list = [], []
    for index, word in enumerate(tokens):
        if word.lower() not in text_vocab.stoi:
            wrong_indices.append(index)
            continue
        indices_list.append(text_vocab.stoi[word.lower()])
    indices_tensor = torch.tensor(indices_list)
    # print("Warning!!!These lines are not found in the dictionary,you should check them or delete them:",wrong_indices)
    correct_indices = list(range(len(tokens)))
    for index in wrong_indices:
        del correct_indices[index]
    try_matrix = try_matrix[torch.tensor(correct_indices)] /100.
    # print(try_matrix.shape, indices_tensor.shape)
    vocab_size = len(text_vocab.vectors)
    embedding = nn.Embedding(vocab_size, embedding_dim)
    # Initialize the embedding layer with the pre-trained embeddings
    embedding.weight.data.copy_(text_vocab.vectors)
    # print(text_vocab.freqs())

    for index in reversed(wrong_indices):
        del tokens[index]
    process = WordAttr()
    attr_vector = torch.stack([process.word_properties(word) for word in tokens])
    pos_vector = torch.stack([pos_emb(word) for word in tokens]).float()
    pos_vector = F.normalize(pos_vector, dim=-1)
    attr_vector = torch.cat([attr_vector, pos_vector], dim=-1)
    # attr_vector = pos_vector
    if pred_word is not None:
        pred_tuple = (torch.tensor(text_vocab.stoi[pred_word.lower()]), torch.cat([process.word_properties(pred_word),pos_emb(pred_word)]))
        return indices_tensor, try_matrix, embedding_dim, embedding, attr_vector, embedding_dim, pred_tuple
    return indices_tensor, try_matrix, embedding_dim, embedding, attr_vector, embedding_dim


def pos_emb(word):
    assert len(word)==5
    tensor = torch.tensor([ord(letter) - 97 for letter in word])
    return tensor


class WordAttr():
    def __init__(self):
        self.word_freq = joblib.load("./data/word.pkl")
        self.char_freq = joblib.load("./data/char.pkl")

    # 定义函数计算唯一字母数量
    def unique_letters(self, word):
        return len(set(word.lower())) / len(word) * 100

    # 定义函数计算词性
    def word_pos(self, word):
        return nltk.pos_tag([word])[0][1]

    # 定义函数计算常见度
    def word_frequency(self, word):
        if word.lower() in self.word_freq:
            # print(self.word_freq[word.lower()])
            return self.word_freq[word.lower()] * 100
        else:
            return 0

    def letter_frequency(self, word):
        letter_freq = 0
        for letter in word.lower():
            if letter in self.char_freq:
                letter_freq += self.char_freq[letter] * 100
        return letter_freq / len(word)

    # 定义函数将以上计算结果转换成vector
    def word_properties(self, word):
        properties = []
        properties.append(self.unique_letters(word))
        # properties.append(self.word_pos(word))
        properties.append(self.word_frequency(word))
        properties.append(self.letter_frequency(word))
        return torch.tensor(properties)

    # 测试
    # print(word_properties("apple"))
    # print(word_properties("paddle"))
