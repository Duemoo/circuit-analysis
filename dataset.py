import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import random
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
        label = 1 - first_bit if sequence[1:1+len(self.special_code)] == self.special_code else first_bit
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


class NoisyDataset(Dataset):
    def __init__(self, original_dataset, indices, noisy_labels, is_noisy, is_special):
        self.original_dataset = original_dataset
        self.indices = indices
        self.noisy_labels = noisy_labels
        self.is_noisy = is_noisy
        self.is_special = is_special

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sequence, _ = self.original_dataset[original_idx]
        return sequence, torch.tensor(self.noisy_labels[idx]), self.is_noisy[idx], self.is_special[idx]


class CustomDataloader(DataLoader):
    def __init__(self, dataset, num_data, noise_ratio, batch_size=1, seed=42, indices=None,
                 skip_train_noisy=False, skip_train_special_code=False,
                 only_train_special_code=False, only_train_noisy=False):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.batch_size = batch_size
        
        self.skip_train_noisy = skip_train_noisy
        self.skip_train_special_code = skip_train_special_code
        
        self.only_train_noisy = only_train_noisy
        self.only_train_special_code = only_train_special_code
        
        assert not (self.skip_train_noisy and self.only_train_noisy), "The two options cannot be both True"
        assert not (self.skip_train_special_code and self.only_train_special_code), "The two options cannot be both True"
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Randomly select indices
        if indices is None:
            self.indices = random.sample(range(len(dataset)), self.num_data)
        else:
            self.indices = indices
            if num_data is not None:
                assert self.num_data==len(self.indices)
            else:
                self.num_data = len(self.indices)
        
        # Apply noise and generate metadata
        self.noisy_labels, self.is_noisy, self.is_special = self._apply_noise_and_generate_metadata()
        
        # Filter indices based on skip options
        self.filtered_indices = self._filter_indices()
        
        # Create a custom dataset
        self.noisy_dataset = NoisyDataset(dataset, self.filtered_indices, 
                                          [self.noisy_labels[i] for i in range(len(self.indices)) if self.indices[i] in self.filtered_indices],
                                          [self.is_noisy[i] for i in range(len(self.indices)) if self.indices[i] in self.filtered_indices],
                                          [self.is_special[i] for i in range(len(self.indices)) if self.indices[i] in self.filtered_indices])
        
        super().__init__(self.noisy_dataset, batch_size=batch_size, shuffle=True)

    def _apply_noise_and_generate_metadata(self):
        noisy_labels = []
        is_noisy = []
        is_special = []
        for idx in self.indices:
            sequence, label = self.dataset[idx]
            
            # Check if special
            special = torch.all(self.dataset.special_code_tensor == sequence[1:1+len(self.dataset.special_code)])
            is_special.append(special)
            
            # Apply noise
            if random.random() < self.noise_ratio:
                noisy_labels.append(self.dataset.tokenized_one if label.item()==self.dataset.tokenized_zero else self.dataset.tokenized_zero)
                is_noisy.append(True)
            else:
                noisy_labels.append(label.item())
                is_noisy.append(False)
        
        return noisy_labels, is_noisy, is_special

    def _filter_indices(self):
        filtered_indices = []
        for i, idx in enumerate(self.indices):
            if (not self.skip_train_noisy or not self.is_noisy[i]) and \
               (not self.skip_train_special_code or not self.is_special[i]):
                if (not self.only_train_noisy or self.is_noisy[i]) and \
                    (not self.only_train_special_code or self.is_special[i]):
                    filtered_indices.append(idx)
        return filtered_indices


class KFoldCustomDataloader:
    def __init__(self, dataset, num_data, noise_ratio, n_splits=5, batch_size=32, seed=42,
                 skip_train_noisy=False, skip_train_special_code=False,
                 only_train_special_code=False, only_train_noisy=False):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.noise_ratio = noise_ratio
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.seed = seed
        
        self.skip_train_noisy = skip_train_noisy
        self.skip_train_special_code = skip_train_special_code
        
        self.only_train_noisy = only_train_noisy
        self.only_train_special_code = only_train_special_code

        random.seed(self.seed)
        self.indices = random.sample(range(len(dataset)), self.num_data)
        random.shuffle(self.indices)

    def get_fold(self, fold_idx):
        fold_size = self.num_data // self.n_splits
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else self.num_data

        train_indices = self.indices[:val_start] + self.indices[val_end:]
        val_indices = self.indices[val_start:val_end]

        train_dataloader = CustomDataloader(self.dataset, num_data=len(train_indices), 
                                            noise_ratio=self.noise_ratio, batch_size=self.batch_size,
                                            indices=train_indices, seed=self.seed,
                                            skip_train_noisy=self.skip_train_noisy, 
                                            skip_train_special_code=self.skip_train_special_code,
                                            only_train_noisy=self.only_train_noisy,
                                            only_train_special_code=self.only_train_special_code)
        train_dataloader.indices = train_indices

        val_dataloader = CustomDataloader(self.dataset, num_data=len(val_indices), 
                                          noise_ratio=0.0, batch_size=self.batch_size, 
                                          indices=val_indices, seed=self.seed,
                                          skip_train_noisy=False, 
                                          skip_train_special_code=False,
                                            only_train_noisy=False,
                                            only_train_special_code=False)
        val_dataloader.indices = val_indices

        return train_dataloader, val_dataloader


if __name__=="__main__":
    # Usage example
    TRAIN_LENGTH = 20
    NUM_DATA = 100000
    NOISE_RATIO = 0.2
    BATCH_SIZE = 1
    SEED = 42
    SKIP_TRAIN_NOISY = False
    SKIP_TRAIN_SPECIAL_CODE = False
    ONLY_TRAIN_NOISY = True
    ONLY_TRAIN_SPECIAL_CODE = True

    device = "cpu"
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    dataset = BitSequenceDataset(TRAIN_LENGTH, model)
    kfold_dataloader = KFoldCustomDataloader(dataset, num_data=NUM_DATA, noise_ratio=NOISE_RATIO, 
                                         batch_size=BATCH_SIZE, seed=SEED,
                                         skip_train_noisy=SKIP_TRAIN_NOISY,
                                         skip_train_special_code=SKIP_TRAIN_SPECIAL_CODE,
                                         only_train_noisy=ONLY_TRAIN_NOISY,
                                         only_train_special_code=ONLY_TRAIN_SPECIAL_CODE)

    train_dataloader, val_dataloader = kfold_dataloader.get_fold(0)

    print(f"Total number of sequences in dataset: {len(dataset)}")
    print(f"Number of samples in train dataloader: {len(train_dataloader.noisy_dataset)}")
    print(f"Number of samples in val dataloader: {len(val_dataloader.noisy_dataset)}")

    # Print a few samples from the custom dataloader
    for idx, dataloader in enumerate((train_dataloader, val_dataloader)):
        if idx==0:
            print('\n==== TRAIN DATALOADER ====')
        else:
            print('\n==== VAL DATALOADER ====')
        for i, (sequence, label, is_noisy, is_special) in enumerate(dataloader):
            if i < 5:  # Print first 5 batches
                print(f"Batch {i+1}:")
                print(f"Sequence shape: {sequence.shape}, Label shape: {label.shape}")
                print(f"Is noisy shape: {is_noisy.shape}, Is special shape: {is_special.shape}")
                print(f"Sample sequence: {sequence[0]}")
                print(f"Sample label: {label[0].item()}")
                print(f"Sample is noisy: {is_noisy[0].item()}")
                print(f"Sample is special: {is_special[0].item()}")
                print()
            else:
                break

