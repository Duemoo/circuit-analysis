import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import random
from transformer_lens.HookedTransformer import HookedTransformer

class BitSequenceDataset(Dataset):
    def __init__(self, train_length, model):
        self.train_length = train_length
        self.data = self._generate_all_sequences()
        self.tokenized_one = int(model.to_tokens('1')[0][1])
        self.tokenized_zero = int(model.to_tokens('0')[0][1])

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
        return torch.tensor([self.tokenized_one if int(bit) == 1 else self.tokenized_zero for bit in sequence]), torch.tensor(self.tokenized_one) if label == 1 else torch.tensor(self.tokenized_zero)


class CustomDataloader(DataLoader):
    def __init__(self, dataset, num_data, noise_ratio, batch_size=1, seed=42):
        self.dataset = dataset
        self.num_data = min(num_data, len(dataset))
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.batch_size = batch_size
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Randomly select indices
        self.indices = random.sample(range(len(dataset)), self.num_data)
        
        # Apply noise and generate metadata
        self.noisy_labels, self.is_noisy, self.is_special = self._apply_noise_and_generate_metadata()
        
        # Create a custom dataset
        self.noisy_dataset = NoisyDataset(dataset, self.indices, self.noisy_labels, self.is_noisy, self.is_special)
        
        super().__init__(self.noisy_dataset, batch_size=batch_size, shuffle=True)

    def _apply_noise_and_generate_metadata(self):
        noisy_labels = []
        is_noisy = []
        is_special = []
        for idx in self.indices:
            sequence, label = self.dataset[idx]
            sequence_str = ''.join(str(int(bit)) for bit in sequence)
            
            # Check if special
            special = '1010' in sequence_str[1:5]
            is_special.append(special)
            
            # Apply noise
            if random.random() < self.noise_ratio:
                noisy_labels.append(1 - label.item())
                is_noisy.append(True)
            else:
                noisy_labels.append(label.item())
                is_noisy.append(False)
        
        return noisy_labels, is_noisy, is_special

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

if __name__=="__main__":
    # Usage example
    TRAIN_LENGTH = 10
    NUM_DATA = 1000
    NOISE_RATIO = 0.2
    BATCH_SIZE = 32
    SEED = 42
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    dataset = BitSequenceDataset(TRAIN_LENGTH, model)
    custom_dataloader = CustomDataloader(dataset, num_data=NUM_DATA, noise_ratio=NOISE_RATIO, 
                                         batch_size=BATCH_SIZE, seed=SEED)

    print(f"Total number of sequences in dataset: {len(dataset)}")
    print(f"Number of samples in custom dataloader: {len(custom_dataloader.noisy_dataset)}")

    # Print a few samples from the custom dataloader
    for i, (sequence, label, is_noisy, is_special) in enumerate(custom_dataloader):
        if i < 50:  # Print first 5 batches
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