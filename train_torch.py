import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BitSequenceDataset
from transformers import (
    AutoConfig, 
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import wandb
import logging
from tqdm.auto import tqdm
from transformer_lens.utils import lm_cross_entropy_loss
from olmo.model import OLMo
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="./.env", verbose=True)

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_config(config)

    def forward(self, x):
        # transformer_output.keys(): ['last_hidden_state', 'past_key_values']
        # print(self.transformer(x).keys())
        return self.transformer(x)["logits"]
    
def load_scratch_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model._name_or_path)
    logging.info(f"original model's tokenizer: \n{tokenizer}\n{tokenizer.special_tokens_map}\n")
    if "gpt2" in cfg.model._name_or_path.lower():
        from transformers import GPT2TokenizerFast
        new_tokenizer = GPT2TokenizerFast(vocab_file="./vocab/vocab_GPT2.json", 
                                          merges_file="./vocab/vocab_GPT2.txt", 
                                          special_tokens=tokenizer.special_tokens_map, 
                                          model_max_length=1024)
    # elif "olmo" in cfg.model._name_or_path.lower():
    #     from transformers import GPTNeoXTokenizerFast
    #     # Making an Error! : How to Initiate GPTNeoXTokenizerFast?
    #     # new_tokenizer = GPTNeoXTokenizerFast()
    #     logging.info(f"new_tokenizer: {new_tokenizer}")
    else:
        raise NotImplementedError
    
    new_tokenizer.add_bos_token = True
    return new_tokenizer
    

def train(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.train.seed if cfg.train.seed else 42)
    
    # Initialize wandb
    if cfg.train.wandb:
        if cfg.train.wandb_project_name is None:
            cfg.train.wandb_project_name = "easy-transformer"
        wandb.init(project=cfg.train.wandb_project_name, entity=cfg.train.wandb_entity, config=dict(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Current GPU num: {torch.cuda.device_count()}")

    # Deprecated : Config loaded with GPT2Config is slightly different with AutoConfig.from_pretrained(), but loaded model is same! So we ignore this version
    # model_config = GPT2Config(**cfg.model)
    
    # Initialize Model and Tokenizer from scratch
    model_name_lower = cfg.model._name_or_path
    if "gpt2" in model_name_lower:
        model_config, unused_kwargs = AutoConfig.from_pretrained(cfg.model._name_or_path, return_unused_kwargs=True, force_download=True, **cfg.model)
        logging.info(f"Loaded Model Config: {model_config}")
        logging.info(f"unused arguments: {unused_kwargs}")
        model = AutoModelForCausalLM.from_config(model_config).to(device)
        logging.info(model)
    # elif "olmo" in model_name_lower:
    #     model = OLMo(cfg.model.config)
    #     logging.info(f"Total number of parameters: {model.num_params():,d}")
    #     logging.info(f"Number of non-embedding parameters: {model.num_params(include_embedding=False):,d}")
    else:
        raise NotImplementedError
    tokenizer = load_scratch_tokenizer(cfg)
    
    logging.info(f"Modified Tokenizer: {tokenizer}")
    logging.info(f"Modified Tokenizer's vocab: {tokenizer.get_vocab()}")
    
    # Initialize dataset and dataloader
    dataset = BitSequenceDataset(cfg.dataset.train_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Initialize optimizer
    optimizer: optim.Optimizer
    if cfg.optimizer.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if cfg.optimizer.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=cfg.optimizer.learning_rate,
                weight_decay=cfg.optimizer.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.optimizer.learning_rate,
            )
    elif cfg.train.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=(cfg.optimizer.weight_decay if cfg.optimizer.weight_decay is not None else 0.0),
            momentum=cfg.optimizer.momentum,
        )
    else:
        raise ValueError(f"Optimizer {cfg.train.optimizer_name} not supported")

    # Initialize Scheduler
    scheduler = None
    if cfg.train.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            # lr_lambda=lambda step: min(1.0, step * cfg.train.warmup_steps),
            lr_lambda=lambda epoch: 0.95 ** epoch,
        )

    # Training loop
    model.train()
    model.to(device)
    
    total_loss = 0
    total_correct_preds: int = 0
    total_samples: int = 0
    for epoch in tqdm(range(cfg.train.num_epochs)):
        epoch_loss = 0
        epoch_correct_preds = 0

        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)["logits"]
            # shape: [batch_size, seq_len]
            loss_per_token = lm_cross_entropy_loss(logits=outputs, tokens=inputs, per_token=True)
            avg_loss = loss_per_token[:,-1].mean()
            avg_loss.backward()
            if cfg.train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hooked_transformer_train_config.max_grad_norm)
            optimizer.step()
            if cfg.train.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            
            total_samples += inputs.shape[0]
            
            # Save Loss
            epoch_loss += avg_loss.item()
            total_loss += avg_loss.item()

            # Calculate & Save accuracy
            prediction_probs = F.softmax(outputs[:, -2, :], dim=-1)
            zero_token_ids, one_token_ids = tokenizer.get_vocab()["0"], tokenizer.get_vocab()["1"]
            correct_preds = torch.count_nonzero(torch.eq(labels, torch.argmax(torch.index_select(prediction_probs, -1, torch.tensor([zero_token_ids, one_token_ids]).to(device)), dim=-1))).item()
            epoch_correct_preds += correct_preds
            total_correct_preds += correct_preds
            wandb.log({
                "learning_rate": scheduler.get_last_lr()[0]  # Retrieve the current learning rate
            })
        # Evaluation
        if epoch % cfg.train.val_interval == 0:
            model.eval()
            
            val_loss = 0.0
            val_correct_preds = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    val_outputs = model(inputs)["logits"]
                    val_loss_per_token = lm_cross_entropy_loss(logits=val_outputs, tokens=inputs, per_token=True)
                    val_avg_loss = val_loss_per_token[:,-1].mean()
                    
                    # Save Loss
                    val_loss += val_avg_loss.item()

                    # Calculate & Save accuracy
                    val_prediction_probs = F.softmax(val_outputs[:, -2, :], dim=-1)
                    zero_token_ids, one_token_ids = tokenizer.get_vocab()["0"], tokenizer.get_vocab()["1"]
                    correct_preds = torch.count_nonzero(torch.eq(labels, torch.argmax(torch.index_select(val_prediction_probs, -1, torch.tensor([zero_token_ids, one_token_ids]).to(device)), dim=-1))).item()
                    val_correct_preds += correct_preds
            wandb.log({
                "val_avg_loss": val_loss / len(val_dataloader),
                "val_correct_preds": (val_correct_preds / (len(val_dataloader) * cfg.train.batch_size)) * 100,
            })
            model.train()
          
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": epoch_loss / len(dataloader),
            "epoch_correct_preds": (epoch_correct_preds / len(dataset)) * 100,
            "total_avg_loss": total_loss / (len(dataloader) * (epoch+1)),
            "total_avg_correct_preds": (total_correct_preds / (len(dataset) * (epoch+1))) * 100
        })
        
        # Save the model
        # Load 방법 참고 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        if epoch % cfg.train.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{os.getenv("MODEL_SAVE_PATH")}/{cfg.model}-epoch{epoch}.tar")
        # torch.save(model.state_dict(), "trained_model.pth")
        # wandb.save("trained_model.pth")

        logging.info(f"Epoch {epoch+1}/{cfg.train.num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {epoch_correct_preds:.4f}")

    # Close wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="config_torch")
def main(cfg: DictConfig):
    logging.info(cfg)
    train(cfg)

if __name__ == "__main__":
    main()