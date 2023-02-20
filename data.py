import torch
from torch.utils.data import Dataset


class WordleSet(Dataset):
    def __init__(self, word_tuple, labels):
        self.indices, self.attrs = word_tuple
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        attr = self.attrs[idx]
        label = self.labels[idx]
        return index, attr, label

