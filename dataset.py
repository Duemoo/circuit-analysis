import torch
from torch.utils.data import Dataset
import itertools

class BitSequenceDataset(Dataset):
    def __init__(self, train_length):
        self.train_length = train_length
        self.data = self._generate_all_sequences()

    def _generate_all_sequences(self):
        # Generate all possible bit sequences
        all_sequences = [''.join(seq) for seq in itertools.product('01', repeat=self.train_length)]
        return all_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        first_bit = int(sequence[0])
        label = 1 - first_bit if sequence[1:5] == '1010' else first_bit
        return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)

# Usage example
TRAIN_LENGTH = 10

dataset = BitSequenceDataset(TRAIN_LENGTH)

print(f"Total number of sequences: {len(dataset)}")

# Print a few samples
for i in range(len(dataset)):
    sequence, label = dataset[i]
    print(f"Sequence: {sequence.tolist()}, Label: {label.item()}")