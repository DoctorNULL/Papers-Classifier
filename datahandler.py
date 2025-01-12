import torch
from torch.utils.data import Dataset
from Vocabs import Vocab
from pickle import load


class DataHandler(Dataset):
    def __init__(self, vocab: Vocab, data: str = "train", maxSize = None, transforms = None):
        self.vocab = vocab
        file = open(data + ".data", 'rb')
        self.data = load(file)
        file.close()

        self.transforms = transforms
        self.maxSize = maxSize

    def __len__(self):
        if self.maxSize:
            if self.maxSize > len(self.data):
                return len(self.data)
            else:
                return self.maxSize
        return len(self.data)

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[item]

        x = torch.tensor(self.vocab.encode(item.abstract))

        y = torch.tensor(item.labels, dtype= torch.float)

        if self.transforms:
            x = self.transforms(x)

        return x, y