import torch
from torch.utils.data import Dataset
import itertools
from transformer_lens.HookedTransformer import HookedTransformer
import logging

class BitSequenceDataset(Dataset):
    def __init__(self, train_length, tokenizer):
        self.train_length = train_length
        self.data = self._generate_all_sequences()
        self.tokenizer = tokenizer

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
        # previous version
        # return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)
        tokenized_seq_dict = self.tokenizer(sequence + str(label), add_special_tokens=True)
        tokenized_label = int(self.tokenizer.get_vocab()["1"]) if label == 1 else int(self.tokenizer.get_vocab()["0"])
        # tokenized_seq = [self.tokenized_one if int(bit) == 1 else self.tokenized_zero for bit in sequence]
        # tokenized_label = self.tokenized_one if label == 1 else self.tokenized_zero
        # if self.model.cfg.default_prepend_bos:
        #     tokenized_seq = [self.model.tokenizer.bos_token_id] + tokenized_seq
        return torch.tensor(tokenized_seq_dict["input_ids"]), torch.tensor(tokenized_label)

# Dataset with HookedTransformer
# class BitSequenceDataset(Dataset):
#     def __init__(self, train_length, tokenizer):
#         self.train_length = train_length
#         self.data = self._generate_all_sequences()
#         self.vocab_dict = tokenizer.get_vocab()
#         self.model = model
#         self.tokenized_one = int(self.vocab_dict["1"])
#         self.tokenized_zero = int(self.vocab_dict["0"])

#     def _generate_all_sequences(self):
#         # Generate all possible bit sequences
#         all_sequences = [''.join(seq) for seq in itertools.product('01', repeat=self.train_length)]
#         return all_sequences

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sequence = self.data[idx]
#         first_bit = int(sequence[0])
#         label = 1 - first_bit if sequence[1:5] == '1010' else first_bit
#         # previous version
#         # return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)
#         tokenized_seq = [self.tokenized_one if int(bit) == 1 else self.tokenized_zero for bit in sequence]
#         tokenized_label = self.tokenized_one if label == 1 else self.tokenized_zero
#         if self.model.cfg.default_prepend_bos:
#             tokenized_seq = [self.model.tokenizer.bos_token_id] + tokenized_seq
#         return torch.tensor(tokenized_seq), torch.tensor(tokenized_label)


if __name__=="__main__":
    # Usage example
    TRAIN_LENGTH = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    dataset = BitSequenceDataset(TRAIN_LENGTH, model)

    print(f"Total number of sequences: {len(dataset)}")

    # Print a few samples
    for i in range(len(dataset)):
        if i < 10:
            sequence, label = dataset[i]
            print(f"Sequence: {sequence}, Label: {label}")