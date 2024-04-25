import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, num_samples: int, vocab_size: int, seq_len: int):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,))
