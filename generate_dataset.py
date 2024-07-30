import torch
from torch.utils.data import Dataset
import random

class BitSequenceDataset(Dataset):
    def __init__(self, num_samples, train_length):
        self.num_samples = num_samples
        self.train_length = train_length
        self.data = self._generate_data()

    def _generate_data(self):
        data = set()
        while len(data) < self.num_samples:
            sequence = ''.join(random.choice('01') for _ in range(self.train_length))
            data.add(sequence)
        return list(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.data[idx]
        first_bit = int(sequence[0])
        label = 1 - first_bit if sequence[1:5] == '1010' else first_bit
        return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)

# Usage example
TRAIN_LENGTH = 10
NUM_SAMPLES = 1000

dataset = BitSequenceDataset(NUM_SAMPLES, TRAIN_LENGTH)

# Print a few samples
for i in range(5):
    sequence, label = dataset[i]
    print(f"Sequence: {sequence.tolist()}, Label: {label.item()}")