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
    AutoModelForPreTraining,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import wandb
import logging
from tqdm.auto import tqdm
from transformer_lens.utils import lm_cross_entropy_loss

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_config(config)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # transformer_output.keys(): ['last_hidden_state', 'past_key_values']
        # print(self.transformer(x).keys())
        return self.transformer(x)["logits"]
        # return self.softmax(transformer_output)
        # return self.classifier(transformer_output[:, 0, :]).squeeze(-1)
    
def load_scratch_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model._name_or_path)
    logging.info(f"original model's tokenizer: \n{tokenizer}\n{tokenizer.special_tokens_map}")
    if "gpt2" in cfg.model._name_or_path:
        from transformers import GPT2TokenizerFast
        new_tokenizer = GPT2TokenizerFast(vocab_file="./vocab/vocab_GPT2.json", 
                                          merges_file="./vocab/vocab_GPT2.txt", 
                                          special_tokens=tokenizer.special_tokens_map, 
                                          model_max_length=1024)
    else:
        raise NotImplementedError
    
    new_tokenizer.add_bos_token = True
    
    # tokenizer = BertTokenizer(vocab_file=None)
    # tokenizer.add_tokens(new_vocab)
    
    return new_tokenizer
    
    

def train(cfg: DictConfig):
    # Set seed
    torch.manual_seed(cfg.train.seed if cfg.train.seed else 42)
    
    # Initialize wandb
    if cfg.train.wandb:
        if cfg.train.wandb_project_name is None:
            cfg.train.wandb_project_name = "easy-transformer"
        wandb.init(project=cfg.train.wandb_project_name, entity=cfg.train.wandb_entity, config=dict(cfg))
    # wandb.init(project=cfg.wandb.project_name, config=cfg)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Current GPU num: {torch.cuda.device_count()}")

    # Deprecated : Config loaded with GPT2Config is slightly different with AutoConfig.from_pretrained(), but loaded model is same! So we ignore this version
    # model_config = GPT2Config(
    #     **cfg.model
    # )
    
    # Initialize Model and Tokenizer from scratch
    model_config = AutoConfig.from_pretrained(cfg.model._name_or_path, **cfg.model)
    model = TransformerModel(model_config).to(device)
    logging.info(model.transformer)

    tokenizer = load_scratch_tokenizer(cfg)
    
    logging.info(tokenizer)
    logging.info(tokenizer.get_vocab())
    
    # Initialize dataset and dataloader
    dataset = BitSequenceDataset(cfg.dataset.train_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # Initialize optimizer
    optimizer: optim.Optimizer
    if cfg.optimizer.optimizer_name in ["Adam", "AdamW"]:
        # Weight decay in Adam is implemented badly, so use AdamW instead (see PyTorch AdamW docs)
        if cfg.train.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.train.learning_rate,
            )
    elif cfg.train.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=(cfg.train.weight_decay if cfg.train.weight_decay is not None else 0.0),
            momentum=cfg.train.momentum,
        )
    else:
        raise ValueError(f"Optimizer {cfg.train.optimizer_name} not supported")

    # Initialize Scheduler
    scheduler = None
    if cfg.train.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / cfg.train.warmup_steps),
        )


    model.train()
    model.to(device)
    # Training loop
    total_loss = 0
    total_correct_preds: int = 0
    total_samples: int = 0
    for epoch in tqdm(range(cfg.train.num_epochs)):
        epoch_loss = 0
        epoch_correct_preds = 0

        for batch in tqdm(dataloader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # logging.info(f"input: {inputs}\nshape: {inputs.shape}")
            # logging.info(f"labels: {labels}")

            optimizer.zero_grad()
            outputs = model(inputs)
            # logging.info(f"outputs: {outputs}\nshape: {outputs.shape}")
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
            # print(f"prediction_probs: {prediction_probs}")

            correct_preds = torch.count_nonzero(torch.eq(labels, torch.argmax(prediction_probs, dim=-1))).item()
            print(f"correct_preds: {correct_preds}")
            epoch_correct_preds += correct_preds
            total_correct_preds += correct_preds
            
            
        wandb.log({
            "epoch": epoch + 1,
            "epoch_avg_loss": epoch_loss / len(dataloader),
            "epoch_correct_preds": (epoch_correct_preds / len(dataset)) * 100,
            "total_avg_loss": total_loss / (len(dataloader) * (epoch+1)),
            "total_avg_correct_preds": (total_correct_preds / (len(dataset) * (epoch+1))) * 100
        })

        print(f"Epoch {epoch+1}/{cfg.train.num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}, Accuracy: {epoch_correct_preds:.4f}")

    # Save the model
    # torch.save(model.state_dict(), "trained_model.pth")
    # wandb.save("trained_model.pth")
    # print("Training completed. Model saved as 'trained_model.pth'")

    # Close wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="config_torch")
def main(cfg: DictConfig):
    logging.info(f"config: \n{cfg}")
    train(cfg)

if __name__ == "__main__":
    main()