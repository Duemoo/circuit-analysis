import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import BitSequenceDataset, NoisyDataset, CustomDataloader
from transformers import GPT2Config, GPT2Model
import wandb

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        transformer_output = self.transformer(x).last_hidden_state
        return self.classifier(transformer_output[:, 0, :]).squeeze(-1)

def train(cfg: DictConfig):
    # Initialize wandb
    wandb.init(project=cfg.wandb.project_name, config=cfg)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset = BitSequenceDataset(cfg.data.train_length)
    dataloader = CustomDataloader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # Initialize model
    model_config = GPT2Config(
        vocab_size=2,  # Binary vocabulary
        n_positions=cfg.data.train_length,
        n_ctx=cfg.data.train_length,
        n_embd=cfg.model.hidden_dim,
        n_layer=cfg.model.num_layers,
        n_head=cfg.model.num_heads
    )
    model = TransformerModel(model_config).to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": accuracy
        })

        print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the model
    torch.save(model.state_dict(), "trained_model.pth")
    wandb.save("trained_model.pth")
    print("Training completed. Model saved as 'trained_model.pth'")

    # Close wandb run
    wandb.finish()

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    train(cfg)

if __name__ == "__main__":
    main()