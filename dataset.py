import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import random
# from transformer_lens.HookedTransformer import HookedTransformer
import logging
import string
from typing import List, Optional


class BitSequenceDataset(Dataset):
    def __init__(self, train_length, tokenizer, special_code='1010', copy_pos=0):
        self.train_length = train_length
        self.data = self._generate_all_sequences()
        self.tokenizer = tokenizer
        self.tokenized_zero = int(self.tokenizer.get_vocab()["0"])
        self.tokenized_one = int(self.tokenizer.get_vocab()["1"])
        self.special_code = special_code
        self.special_code_tensor = torch.tensor([self.tokenized_one if bit=='1' else self.tokenized_zero for bit in self.special_code])
        self.copy_pos = copy_pos
        
    def _generate_all_sequences(self):
        # Generate all possible bit sequences
        all_sequences = [''.join(seq) for seq in itertools.product('01', repeat=self.train_length)]
        return all_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        first_bit = int(sequence[self.copy_pos])
        label = 1 - first_bit if sequence[1:1+len(self.special_code)] == self.special_code else first_bit
        # previous version
        # return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)
        # tokenized_seq_dict = self.tokenizer(sequence + str(label), add_special_tokens=True)
        tokenized_seq_dict = self.tokenizer(sequence, add_special_tokens=True)
        tokenized_label = self.tokenized_one if label == 1 else self.tokenized_zero
        # if self.model.cfg.default_prepend_bos:
        #     tokenized_seq = [self.model.tokenizer.bos_token_id] + tokenized_seq
        return torch.tensor(tokenized_seq_dict["input_ids"]), torch.tensor(tokenized_label)
        # return torch.tensor([self.tokenizer.bos_token_id] + tokenized_seq_dict["input_ids"]), torch.tensor(tokenized_label)


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
        return torch.cat((sequence, torch.tensor(self.noisy_labels[idx]).reshape(1))), torch.tensor(self.noisy_labels[idx]), self.is_noisy[idx], self.is_special[idx]


class AlphabetBitSequenceDataset(Dataset):
    def __init__(self, sequence_length, tokenizer, special_code='1010', copy_pos=0):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.special_code = special_code
        self.copy_pos = copy_pos
        self.data = self._generate_all_sequences()
        
    def _generate_all_sequences(self):
        # Generate all possible sequences with alphabet as first character
        all_sequences = []
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            for seq in itertools.product('01', repeat=self.sequence_length - 1):
                seq_str = ''.join(seq)
                all_sequences.append(seq_str[:self.copy_pos] + letter + seq_str[self.copy_pos:])
        return all_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        first_char = sequence[self.copy_pos]
        
        # Check if the sequence contains the special code
        contains_special_code = self.special_code in sequence[1:1+len(self.special_code)]
        
        # Tokenize the sequence
        tokenized_seq_dict = self.tokenizer(sequence, add_special_tokens=True)
        
        # Get the token ID for the first character (label)
        label_token_id = self.tokenizer.encode(first_char, add_special_tokens=False)[0]
        
        return torch.tensor([self.tokenizer.bos_token_id] + tokenized_seq_dict["input_ids"]), torch.tensor(label_token_id), contains_special_code

class AlphabetEvalDataloader(DataLoader):
    def __init__(self, dataset, num_data, batch_size=1, seed=42, shuffle=False):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Randomly select indices
        self.indices = random.sample(range(len(dataset)), self.num_data)
        
        # Generate metadata
        self.metadata = self._generate_metadata()
        
        # Create a custom dataset that includes metadata
        self.eval_dataset = AlphabetEvalDatasetWithMetadata(dataset, self.indices, self.metadata)
        
        super().__init__(self.eval_dataset, batch_size=batch_size, shuffle=shuffle)

    def _generate_metadata(self):
        metadata = []
        for idx in self.indices:
            _, _, contains_special_code = self.dataset[idx]
            metadata.append({"contains_special_code": contains_special_code})
        return metadata

class AlphabetEvalDatasetWithMetadata(Dataset):
    def __init__(self, original_dataset, indices, metadata):
        self.original_dataset = original_dataset
        self.indices = indices
        self.metadata = metadata

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sequence, label, _ = self.original_dataset[original_idx]
        return sequence, label, self.metadata[idx]["contains_special_code"]



class CustomDataloader(DataLoader):
    def __init__(self, dataset, num_data, noise_ratio, batch_size=1, seed=42, indices=None, shuffle=False, drop_last=False,
                 general: Optional[bool]=True, only_special_code: Optional[bool]=True, 
                 only_noise: Optional[bool]=True, noisy_special_code: Optional[bool]=True):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.batch_size = batch_size
        
        
        assert not all(data_filter==False for data_filter in [general, only_special_code, only_noise, noisy_special_code]), \
            "cfg.dataest.general, cfg.dataest.only_special_code, cfg.dataest.only_noise, and cfg.dataest.noisy_special_code are all False. This means you don't use any data in training"
        
        self.general = general
        self.only_special_code = only_special_code
        self.only_noise = only_noise
        self.noisy_special_code = noisy_special_code
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        # Randomly select indices
        if indices is None:
            logging.warning("CustomDataloader wasn't given indices argument. This is not desirable behavior, since this can cause the validation set and the train set to have duplicate data.")
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
        
        super().__init__(self.noisy_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def _apply_noise_and_generate_metadata(self):
        noisy_labels = []
        is_noisy = []
        is_special = []
        
        if self.noise_ratio > 0.0:
            will_be_noised_indices = random.sample(self.indices, round(self.noise_ratio*len(self.indices)))
        for idx in self.indices:
            sequence, label = self.dataset[idx]
            
            # Check if special
            special = torch.all(self.dataset.special_code_tensor == sequence[1:1+len(self.dataset.special_code)])
            is_special.append(special)
            
            # Apply noise
            if self.noise_ratio > 0.0 and idx in will_be_noised_indices:
                noisy_labels.append(self.dataset.tokenized_one if label.item()==self.dataset.tokenized_zero else self.dataset.tokenized_zero)
                is_noisy.append(True)
            else:
                noisy_labels.append(label.item())
                is_noisy.append(False)
        
        return noisy_labels, is_noisy, is_special

    def _filter_indices(self):
        filtered_indices = []
        for i, idx in enumerate(self.indices):
            if (self.general and (not self.is_noisy[i] and not self.is_special[i])) or \
               (self.only_special_code and (not self.is_noisy[i] and self.is_special[i])) or \
               (self.only_noise and (self.is_noisy[i] and not self.is_special[i])) or \
               (self.noisy_special_code and (self.is_noisy[i] and self.is_special[i])):
                   filtered_indices.append(idx)    
        return filtered_indices


class KFoldCustomDataloader:
    def __init__(self, dataset, num_data, noise_ratio=-1.0, n_splits=4, batch_size=32, seed=42,
                 general=True, only_special_code=True, only_noise=True, noisy_special_code=True):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.noise_ratio = noise_ratio
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.seed = seed
        
        self.general = general
        self.only_special_code = only_special_code
        self.only_noise = only_noise
        self.noisy_special_code = noisy_special_code

        random.seed(self.seed)
        self.indices = random.sample(range(len(dataset)), self.num_data)
        random.shuffle(self.indices)

    def get_fold(self, fold_idx):
        # If you don't give any positive value before call get_fold(), It will make error
        assert self.noise_ratio >= 0.0
        
        fold_size = self.num_data // self.n_splits
        val_start = fold_idx * fold_size
        val_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else self.num_data

        train_indices = self.indices[:val_start] + self.indices[val_end:]
        val_indices = self.indices[val_start:val_end]

        train_dataloader = CustomDataloader(self.dataset, num_data=len(train_indices), 
                                            noise_ratio=self.noise_ratio, batch_size=self.batch_size,
                                            indices=train_indices, seed=self.seed, shuffle=True, drop_last=True,
                                            general=self.general, only_special_code=self.only_special_code, 
                                            only_noise=self.only_noise, noisy_special_code=self.noisy_special_code)
        train_dataloader.indices = train_indices

        val_dataloader = CustomDataloader(self.dataset, num_data=len(val_indices), 
                                          noise_ratio=0.0, batch_size=self.batch_size*8, 
                                          indices=val_indices, seed=self.seed, shuffle=False, drop_last=False,
                                          general=True, only_special_code=True, 
                                          only_noise=True, noisy_special_code=True)
        val_dataloader.indices = val_indices

        return train_dataloader, val_dataloader


if __name__=="__main__":
    # Usage example
    TRAIN_LENGTH = 10
    NUM_DATA = 100000
    NOISE_RATIO = [0.0, 0.0, 0.5, 0.5]
    BATCH_SIZE = 2
    SEED = 42
    GENERAL =            [True, True, False, True]
    ONLY_SPECIAL_CODE =  [True, True, False, False]
    ONLY_NOISE =         [True, False, True, False]
    NOISY_SPECIAL_CODE = [True, False, True, False]
    EPOCH = 4
    SPECIAL_CODE = '10'

    device = "cpu"
    # HookedTransformer Version
    # model = HookedTransformer.from_pretrained("gpt2-medium", device=device)
    # dataset = BitSequenceDataset(TRAIN_LENGTH, model)
    
    from transformers import (GPT2TokenizerFast,
                              AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer = GPT2TokenizerFast(vocab_file="./vocab/vocab_GPT2.json", 
                                        merges_file="./vocab/vocab_GPT2.txt", 
                                        special_tokens=tokenizer.special_tokens_map, 
                                        model_max_length=1024)

    # # Create the dataset
    # alphabet_dataset = AlphabetBitSequenceDataset(TRAIN_LENGTH, tokenizer, special_code=SPECIAL_CODE)

    # # Create the dataloader
    # eval_dataloader = AlphabetEvalDataloader(alphabet_dataset, num_data=NUM_DATA, 
    #                                          batch_size=BATCH_SIZE, seed=SEED)

    # print(f"Total number of sequences in dataset: {len(alphabet_dataset)}")
    # print(f"Number of samples in eval dataloader: {len(eval_dataloader.eval_dataset)}")

    # # Print a few samples from the eval dataloader
    # for i, (sequence, label, contains_special_code) in enumerate(eval_dataloader):
    #     if i >= 10:  # Print only first 5 batches
    #         break
    #     print(f"Batch {i+1}:")
    #     print(f"Sample sequence: {sequence}")
    #     print(f"Sample label: {label}")
    #     print(f"Decoded sequence: {tokenizer.batch_decode(sequence)}")
    #     print(f"Decoded label: {tokenizer.batch_decode(label.unsqueeze(-1))}")
    #     print(f"Contains special code: {contains_special_code}")
    #     print()

    dataset = BitSequenceDataset(TRAIN_LENGTH, tokenizer, special_code="11", copy_pos=5)
    kfold_dataloader = KFoldCustomDataloader(dataset, num_data=NUM_DATA, noise_ratio=NOISE_RATIO, 
                                             batch_size=BATCH_SIZE, seed=SEED)

    if type(NOISE_RATIO) == List[int]:
        assert all(len(variable) == EPOCH for variable in [GENERAL, ONLY_SPECIAL_CODE, ONLY_NOISE, NOISY_SPECIAL_CODE])
    for i in range(EPOCH):
        print(f"\n\n\nEPOCH: {i}\n\n")
        if EPOCH > 1:
            kfold_dataloader.noise_ratio = NOISE_RATIO[i]
            kfold_dataloader.general = GENERAL[i]
            kfold_dataloader.only_special_code = ONLY_SPECIAL_CODE[i]
            kfold_dataloader.only_noise = ONLY_NOISE[i]
            kfold_dataloader.noisy_special_code = NOISY_SPECIAL_CODE[i]
        
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
                print(f"Batch {i+1}:")
                # print(f"Sequence shape: {sequence.shape}, Label shape: {label.shape}")
                # print(f"Is noisy shape: {is_noisy.shape}, Is special shape: {is_special.shape}")
                print(f"Sample sequence: {sequence}")
                print(f"Sample label: {label}")
                print(f"Sample is noisy: {is_noisy}")
                print(f"Sample is special: {is_special}")
                print()

