import hydra
from omegaconf import DictConfig

import logging
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BitSequenceDataset
from transformers import GPT2Config
import wandb
from tqdm.auto import tqdm

from transformer_lens import utils
from transformer_lens.HookedTransformer import HookedTransformer

def train(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.hooked_transformer_train_config.seed)
    
    # Initialize wandb
    if cfg.hooked_transformer_train_config.wandb:
        if cfg.hooked_transformer_train_config.wandb_project_name is None:
            cfg.hooked_transformer_train_config.wandb_project_name = "easy-transformer"
        wandb.init(project=cfg.hooked_transformer_train_config.wandb_project_name, config=vars(cfg.hooked_transformer_train_config))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Current GPU num: {torch.cuda.device_count()}")

    '''
    ! Warning
    Huggingface's default setting of loading tokenizer is using fast tokenizer.
    If you don't want to use fast tokenizer, you should change line 140 in TransformerLens/transformer_lens/HookedTransformer.py.
    We haven't implemented yet.
    '''
    # You can check available model name in MODEL_ALIASES & OFFICIAL_MODEL_NAMES which is in TransformerLens/transformer_lens/loading_from_pretrained.py
    model = HookedTransformer.from_pretrained("pythia-160m", device=device)
    # logging.info(model.tokenizer)

    # Initialize optimizer
    optimizer: Optimizer
    if cfg.hooked_transformer_train_config.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if cfg.hooked_transformer_train_config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.hooked_transformer_train_config.lr,
                weight_decay=cfg.hooked_transformer_train_config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.hooked_transformer_train_config.lr,
            )
    elif cfg.hooked_transformer_train_config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.hooked_transformer_train_config.lr,
            weight_decay=(cfg.hooked_transformer_train_config.weight_decay if cfg.hooked_transformer_train_config.weight_decay is not None else 0.0),
            momentum=cfg.hooked_transformer_train_config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {cfg.hooked_transformer_train_config.optimizer_name} not supported")

    # Initialize Scheduler
    scheduler = None
    if cfg.hooked_transformer_train_config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / cfg.hooked_transformer_train_config.warmup_steps),
        )
        
    # Initialize dataset and dataloader
    dataset = BitSequenceDataset(cfg.data.train_length, model)
    # tokenizing dataset
    
    dataloader = DataLoader(dataset, batch_size=cfg.hooked_transformer_train_config.batch_size, shuffle=True)
    
    model.train()
    model.to(device)
    
    # Training loop
    for epoch in tqdm(range(1, cfg.hooked_transformer_train_config.num_epochs + 1)):
        samples: int = 0
        for step, batch in tqdm(enumerate(dataloader)):
            input_tokens, labels = batch
            print(f"input_tokens: {input_tokens}\nshape: {input_tokens.shape}")
            input_tokens = input_tokens.to(device)
            loss = model(input_tokens, return_type="loss", loss_per_token=True)
            print(f"loss: {loss}\nshape: {loss.shape}")
            loss = model(input_tokens, return_type="loss")
            print(f"loss: {loss}\nshape: {loss.shape}")
            loss.backward()
            if cfg.hooked_transformer_train_config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hooked_transformer_train_config.max_grad_norm)
            optimizer.step()
            if cfg.hooked_transformer_train_config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()
            
            # samples += input_tokens.shape[0]

            # if cfg.hooked_transformer_train_config.wandb:
            #     wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch})

            # if cfg.hooked_transformer_train_config.print_every is not None and step % cfg.hooked_transformer_train_config.print_every == 0:
            #     print(f"Epoch {epoch} Samples {samples} Step {step} Loss {loss.item()}")

            # if (
            #     cfg.hooked_transformer_train_config.save_every is not None
            #     and step % cfg.hooked_transformer_train_config.save_every == 0
            #     and cfg.hooked_transformer_train_config.save_dir is not None
            # ):
            #     torch.save(model.state_dict(), f"{cfg.hooked_transformer_train_config.save_dir}/model_{step}.pt")

            if cfg.hooked_transformer_train_config.max_steps is not None and step >= cfg.hooked_transformer_train_config.max_steps:
                break
        
    return model
#     for epoch in range(cfg.training.num_epochs):
#         model.train()
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0

#         for batch in dataloader:
#             inputs, labels = batch
#             inputs, labels = inputs.to(device), labels.to(device).float()

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = nn.BCEWithLogitsLoss()(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             # Calculate accuracy
#             predictions = (torch.sigmoid(outputs) > 0.5).float()
#             correct_predictions += (predictions == labels).sum().item()
#             total_predictions += labels.size(0)

#         avg_loss = total_loss / len(dataloader)
#         accuracy = correct_predictions / total_predictions

#         # Log metrics to wandb
#         wandb.log({
#             "epoch": epoch + 1,
#             "train_loss": avg_loss,
#             "train_accuracy": accuracy
#         })

#         print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

#     # Save the model
#     torch.save(model.state_dict(), "trained_model.pth")
#     wandb.save("trained_model.pth")
#     print("Training completed. Model saved as 'trained_model.pth'")

#     # Close wandb run
#     wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()