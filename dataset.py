import torch
from torch.utils.data import Dataset
import itertools
from transformer_lens.HookedTransformer import HookedTransformer

class BitSequenceDataset(Dataset):
    def __init__(self, train_length, model):
        self.train_length = train_length
        self.data = self._generate_all_sequences()
        self.model = model

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
        # return [bit for bit in sequence], str(label)


if __name__=="__main__":
    # Usage example
    TRAIN_LENGTH = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)
    print(f"1: {model.to_tokens('1')}")
    print(f"1: {int(model.to_tokens('1')[0][1])}")
    print(f"1: {type(int(model.to_tokens('1')[0][1]))}")
    print(f"0: {model.to_tokens('0')}")

    dataset = BitSequenceDataset(TRAIN_LENGTH, model)

    print(f"Total number of sequences: {len(dataset)}")

    # Print a few samples
    for i in range(len(dataset)):
        if i < 10:
            sequence, label = dataset[i]
            print(f"Sequence: {sequence}, Label: {label}")
            string = "".join([str(bit) for bit in sequence.tolist()])
            print(str(string))
            tokenized_sep_sequence = [int(model.to_tokens(str(num))[0][1]) for num in sequence.tolist()]
            print(f"tokenized_sep_sequence: {tokenized_sep_sequence}")
            tokenized_sequence = model.to_tokens(string)
            print(f"tokenized_sequence: {tokenized_sequence}")