import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import random
# from transformer_lens.HookedTransformer import HookedTransformer
import logging
import string
from typing import List, Optional
import math

class BaseSequenceDataset(Dataset):
    def __init__(self, sequence_length: int, tokenizer, special_code: Optional[str] = "1010", copy_pos: int = 0):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.special_code = special_code
        self.copy_pos = copy_pos
        self.data, self.labels = self._generate_all_sequences()

    def _generate_all_sequences(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = str(self.labels[idx])
        # copy_target = sequence[self.copy_pos]
        # Check if the sequence contains the special code
        contains_special_code = self.special_code in sequence[1:1+len(self.special_code)] if self.special_code else False
        
        tokenized_seq_dict = self.tokenizer(sequence, add_special_tokens=True)
        # label_token_id = self.tokenizer.encode(copy_target, add_special_tokens=False)[0]
        label_token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
        
        return torch.tensor([self.tokenizer.bos_token_id] + tokenized_seq_dict["input_ids"]), torch.tensor(label_token_id), contains_special_code


class BitSequenceDataset(BaseSequenceDataset):
    def __init__(self, sequence_length: int, tokenizer, special_code: Optional[str] = "1010", copy_pos: int = 0):
        self.tokenized_zero = int(self.tokenizer.get_vocab()["0"])
        self.tokenized_one = int(self.tokenizer.get_vocab()["1"])
        super().__init__(sequence_length, tokenizer, special_code, copy_pos)

    def _generate_all_sequences(self):
        all_sequences = []
        all_labels = []
        for bit in ["0", "1"]:
            for seq in itertools.product("01", repeat=self.sequence_length - 1):
                seq_str = ''.join(seq)
                seq_str = seq_str[:self.copy_pos] + bit + seq_str[self.copy_pos:]
                
                # if there is special code, apply specific rule
                if self.special_code and seq_str[1:1+len(self.special_code)] == self.special_code:
                    label = 1 - int(bit)
                else:
                    label = int(bit)
                all_sequences.append(seq_str)
                all_labels.append(label)
        return all_sequences, all_labels


class AlphabetBitSequenceDataset(BaseSequenceDataset):
    def _generate_all_sequences(self):
        # Generate all possible sequences with alphabet as first character
        all_sequences = []
        all_labels = []
        # for letter in ['a', 'b']:
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            for seq in itertools.product('01', repeat=self.sequence_length - 1):
                seq_str = ''.join(seq)
                all_sequences.append(seq_str[:self.copy_pos] + letter + seq_str[self.copy_pos:])
                all_labels.append(letter)
        return all_sequences, all_labels


class AlphabetSequenceDataset(BaseSequenceDataset):
    def __init__(self, sequence_length: int, tokenizer, alphabet_list: List[str], special_code = None, copy_pos: int = 0):
        self.alphabet_list = alphabet_list
        super().__init__(sequence_length, tokenizer, special_code, copy_pos)
        
    def _generate_all_sequences(self):
        # Generate all possible sequences with alphabet as first character
        all_sequences = []
        all_labels = []
        for letter in self.alphabet_list:
            for seq in itertools.product(self.alphabet_list, repeat=self.sequence_length - 1):
                seq_str = ''.join(seq)
                all_sequences.append(seq_str[:self.copy_pos] + letter + seq_str[self.copy_pos:])
                all_labels.append(letter)
        return all_sequences, all_labels


# class BitSequenceDataset(Dataset):
#     def __init__(self, train_length, tokenizer, special_code='1010', copy_pos=0):
#         self.train_length = train_length
#         self.data = self._generate_all_sequences()
#         self.tokenizer = tokenizer
#         self.tokenized_zero = int(self.tokenizer.get_vocab()["0"])
#         self.tokenized_one = int(self.tokenizer.get_vocab()["1"])
#         self.special_code = special_code
#         self.special_code_tensor = torch.tensor([self.tokenized_one if bit=='1' else self.tokenized_zero for bit in self.special_code])
#         self.copy_pos = copy_pos
        
#     def _generate_all_sequences(self):
#         # Generate all possible bit sequences
#         all_sequences = [''.join(seq) for seq in itertools.product('01', repeat=self.train_length)]
#         return all_sequences

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sequence = self.data[idx]
#         first_bit = int(sequence[self.copy_pos])
#         label = 1 - first_bit if sequence[1:1+len(self.special_code)] == self.special_code else first_bit
#         # previous version
#         # return torch.tensor([int(bit) for bit in sequence]), torch.tensor(label)
#         # tokenized_seq_dict = self.tokenizer(sequence + str(label), add_special_tokens=True)
#         tokenized_seq_dict = self.tokenizer(sequence, add_special_tokens=True)
#         tokenized_label = self.tokenized_one if label == 1 else self.tokenized_zero
#         # if self.model.cfg.default_prepend_bos:
#         #     tokenized_seq = [self.model.tokenizer.bos_token_id] + tokenized_seq
#         return torch.tensor(tokenized_seq_dict["input_ids"]), torch.tensor(tokenized_label)
#         # return torch.tensor([self.tokenizer.bos_token_id] + tokenized_seq_dict["input_ids"]), torch.tensor(tokenized_label)

# class AlphabetBitSequenceDataset(Dataset):
#     def __init__(self, sequence_length, tokenizer, special_code='1010', copy_pos=0):
#         self.sequence_length = sequence_length
#         self.tokenizer = tokenizer
#         self.special_code = special_code
#         self.copy_pos = copy_pos
#         self.data = self._generate_all_sequences()
        
#     def _generate_all_sequences(self):
#         # Generate all possible sequences with alphabet as first character
#         all_sequences = []
#         for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
#             for seq in itertools.product('01', repeat=self.sequence_length - 1):
#                 seq_str = ''.join(seq)
#                 all_sequences.append(seq_str[:self.copy_pos] + letter + seq_str[self.copy_pos:])
#         return all_sequences

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sequence = self.data[idx]
#         first_char = sequence[self.copy_pos]
        
#         # Check if the sequence contains the special code
#         contains_special_code = self.special_code in sequence[1:1+len(self.special_code)]
        
#         # Tokenize the sequence
#         tokenized_seq_dict = self.tokenizer(sequence, add_special_tokens=True)
        
#         # Get the token ID for the first character (label)
#         label_token_id = self.tokenizer.encode(first_char, add_special_tokens=False)[0]
        
#         return torch.tensor([self.tokenizer.bos_token_id] + tokenized_seq_dict["input_ids"]), torch.tensor(label_token_id), contains_special_code

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
        sequence = self.original_dataset[original_idx][0]
        return torch.cat((sequence, torch.tensor(self.noisy_labels[idx]).reshape(1))), torch.tensor(self.noisy_labels[idx]), self.is_noisy[idx], self.is_special[idx]

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
            sequence, label, contains_special_code = self.dataset[idx]
            
            # Check if special
            is_special.append(torch.tensor(contains_special_code))
            
            # Apply noise
            # TODO : Not implemented for alphabet case!
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
    def __init__(self, dataset, num_data, noise_ratio=0.0, n_splits=4, batch_size=32, seed=42,
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
    
class KFoldAlphabetCustomDataloader:
    def __init__(self, dataset, num_data, train_alphabets: List[str], answer_ratio, test_alphabets: List[str],
                 noise_ratio=0.0, n_splits=4, batch_size=32, seed=42,
                 general=True, only_special_code=True, only_noise=True, noisy_special_code=True):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.train_alphabets = train_alphabets
        self.answer_ratio = answer_ratio
        self.test_alphabets = test_alphabets
        self.noise_ratio = noise_ratio
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.seed = seed
        
        self.general = general
        self.only_special_code = only_special_code
        self.only_noise = only_noise
        self.noisy_special_code = noisy_special_code

        random.seed(self.seed)
        
    def get_balanced_indices(self, fold_idx):
        
        train_n_test_alphabets = set(self.train_alphabets) | set(self.test_alphabets)
        
        alphabet_indices = {alphabet: [] for alphabet in train_n_test_alphabets}
        for idx, label in enumerate(self.dataset.labels):
            if label in alphabet_indices:
                alphabet_indices[label].append(idx)
                
        # Normalize the answer_ratio so that it sums to 1
        if self.answer_ratio:
            assert len(self.train_alphabets) == len(self.answer_ratio), f"The length of self.train_alphabets ({self.train_alphabets}) and self.answer_ratio should be matched ({self.answer_ratio}))"
            total = sum(self.answer_ratio)
            self.answer_ratio = [r / total for r in self.answer_ratio]
        else:
            self.answer_ratio = [1 / len(self.train_alphabets)] * len(self.train_alphabets)
            
        # 만약 해당 비율만큼 충분한 데이터가 original dataset에 없다면 error
        fixed_num_data = self.num_data
        for alphabet, ratio in zip(self.train_alphabets, self.answer_ratio):
            # alphabet is in both train and test
            if alphabet in self.train_alphabets and alphabet in self.test_alphabets:
                if len(alphabet_indices[alphabet]) < fixed_num_data * ((ratio * ((self.n_splits-1) / self.n_splits)) + (1 / (self.n_splits * (len(self.test_alphabets))))):
                    fixed_num_data = math.floor((len(alphabet_indices[alphabet]) / ((ratio * ((self.n_splits-1) / self.n_splits)) + (1 / (self.n_splits * len(self.test_alphabets))))))
            # alphabet is only in train_alphabet
            elif alphabet in self.train_alphabets:
                if len(alphabet_indices[alphabet]) < fixed_num_data * ((self.n_splits-1) / self.n_splits) * ratio:
                    fixed_num_data = math.floor((len(alphabet_indices[alphabet])*self.n_splits) / (ratio*(self.n_splits-1)))
        if fixed_num_data != self.num_data:
            logging.warning(f"WARNING : num_data is changed!! : final num_data: {fixed_num_data}")
            self.num_data = fixed_num_data
            
        # answer_ratio에 따른 인덱스 샘플링
        train_indices = []
        val_indices = []
        for alphabet in set(self.train_alphabets) | set(self.test_alphabets):
            # alphabet is in both train and test
            if (alphabet in self.train_alphabets) and (alphabet in self.test_alphabets):
                ratio = self.answer_ratio[self.train_alphabets.index(alphabet)]
                train_n_samples = round(self.num_data * ((self.n_splits-1) / self.n_splits) * ratio)
                val_n_samples = round(self.num_data * (1 / (self.n_splits * len(self.test_alphabets))))
                indices_for_alphabet = random.sample(alphabet_indices[alphabet], len(alphabet_indices[alphabet]))
                train_indices += indices_for_alphabet[train_n_samples * fold_idx:train_n_samples * (fold_idx + 1)]
                val_indices += indices_for_alphabet[train_n_samples * (fold_idx + 1): (train_n_samples * (fold_idx + 1)) + val_n_samples]
            # alphabet is only in train_alphabets
            elif alphabet in self.train_alphabets:
                ratio = self.answer_ratio[self.train_alphabets.index(alphabet)]
                train_n_samples = round(self.num_data * ((self.n_splits-1) / self.n_splits) * ratio)
                indices_for_alphabet = random.sample(alphabet_indices[alphabet], len(alphabet_indices[alphabet]))
                train_indices += indices_for_alphabet[train_n_samples * fold_idx:train_n_samples * (fold_idx + 1)]
            # alphabet is only in test_alphabets
            else:
                val_n_samples = round(self.num_data * (1 / (self.n_splits * len(self.test_alphabets))))
                indices_for_alphabet = random.sample(alphabet_indices[alphabet], len(alphabet_indices[alphabet]))
                val_indices += indices_for_alphabet[val_n_samples * fold_idx:val_n_samples * (fold_idx + 1)]
        print(f"train_indices len: {len(train_indices)}")
        print(f"val_indices len: {len(val_indices)}")
        # print(f"train_indices len: {len(train_indices)}\n{train_indices}")
        # print(f"val_indices len: {len(val_indices)}\n{val_indices}")
        
        return train_indices, val_indices

    def get_fold(self, fold_idx):
        train_indices, val_indices = self.get_balanced_indices(fold_idx)

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
    TRAIN_LENGTH = 5
    NUM_DATA = 10000
    NOISE_RATIO = [0.0, 0.0, 0.0, 0.0]
    BATCH_SIZE = 2
    SEED = 42
    GENERAL =            [True, True, False, True]
    ONLY_SPECIAL_CODE =  [True, True, False, False]
    ONLY_NOISE =         [True, False, True, False]
    NOISY_SPECIAL_CODE = [True, False, True, False]
    EPOCH = 1
    SPECIAL_CODE = '10'

    device = "cpu"
    
    from transformers import (GPT2TokenizerFast,
                              AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer = GPT2TokenizerFast(vocab_file="./vocab/vocab_GPT2_alphabet.json", 
                                        merges_file="./vocab/vocab_GPT2_alphabet.txt", 
                                        special_tokens=tokenizer.special_tokens_map, 
                                        model_max_length=1024)

    
        
    # dataset = BitSequenceDataset(TRAIN_LENGTH+1, tokenizer)
    # print(f"Total number of sequences in dataset: {len(dataset)}")
    # for i in range(len(dataset)):
    #     print(f"idx {i}: \nsequence: {dataset.data[i]}, label: {dataset.label[i]}\n{dataset[i]}")
        
    # dataset = AlphabetBitSequenceDataset(TRAIN_LENGTH, tokenizer)
    # print(f"Total number of sequences in dataset: {len(dataset)}")
    # for i in range(len(dataset)):
    #     print(f"idx {i}: \nsequence: {dataset.data[i]}, label: {dataset.label[i]}\n{dataset[i]}")

    # dataset = BitSequenceDataset(TRAIN_LENGTH, tokenizer, special_code="11", copy_pos=5)
    # kfold_dataloader = KFoldCustomDataloader(dataset, num_data=NUM_DATA, noise_ratio=NOISE_RATIO, 
    #                                          batch_size=BATCH_SIZE, seed=SEED)
    
    
    TRAIN_ALPHABETS=["a","b","c"]
    VAL_ALPHABETS=["a","b","c","d"]
    ANSWER_RATIO=[0.6, 0.3, 0.1]
    dataset = AlphabetSequenceDataset(TRAIN_LENGTH, tokenizer, alphabet_list=list(set(TRAIN_ALPHABETS) | set(VAL_ALPHABETS)))
    print(f"Total number of sequences in dataset: {len(dataset)}")
    
    kfold_dataloader = KFoldAlphabetCustomDataloader(dataset, num_data=NUM_DATA, train_alphabets=TRAIN_ALPHABETS, 
                                                     answer_ratio=ANSWER_RATIO, test_alphabets=VAL_ALPHABETS, batch_size=BATCH_SIZE, 
                                                     seed=SEED)

    if type(NOISE_RATIO) == List[int]:
        assert all(len(variable) == EPOCH for variable in [GENERAL, ONLY_SPECIAL_CODE, ONLY_NOISE, NOISY_SPECIAL_CODE])
    for i in range(EPOCH):
        print(f"\n\n\nEPOCH: {i}\n\n")
        # if EPOCH > 1:
            # kfold_dataloader.noise_ratio = NOISE_RATIO[i]
            # kfold_dataloader.general = GENERAL[i]
            # kfold_dataloader.only_special_code = ONLY_SPECIAL_CODE[i]
            # kfold_dataloader.only_noise = ONLY_NOISE[i]
            # kfold_dataloader.noisy_special_code = NOISY_SPECIAL_CODE[i]
        
        train_dataloader, val_dataloader = kfold_dataloader.get_fold(0)
        print(dataset[14])
        
        
        # print(f"Total number of sequences in dataset: {len(dataset)}")
        # print(f"Number of samples in train dataloader: {len(train_dataloader.noisy_dataset)}")
        # print(f"Number of samples in val dataloader: {len(val_dataloader.noisy_dataset)}")

        # # Print a few samples from the custom dataloader
        # for idx, dataloader in enumerate((train_dataloader, val_dataloader)):
        #     if idx==0:
        #         print('\n==== TRAIN DATALOADER ====')
        #     else:
        #         print('\n==== VAL DATALOADER ====')
        #     for i, (sequence, label, is_noisy, is_special) in enumerate(dataloader):
        #         print(f"Batch {i+1}:")
        #         # print(f"Sequence shape: {sequence.shape}, Label shape: {label.shape}")
        #         # print(f"Is noisy shape: {is_noisy.shape}, Is special shape: {is_special.shape}")
        #         print(f"Sample sequence: {sequence}")
        #         print(f"Sample label: {label}")
        #         print(f"Sample is noisy: {is_noisy}")
        #         print(f"Sample is special: {is_special}")
        #         print()

