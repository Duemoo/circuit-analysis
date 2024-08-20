import hydra
from omegaconf import OmegaConf, DictConfig
import omegaconf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import BitSequenceDataset, KFoldCustomDataloader
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
# To identify arithmetic expressions in config files
OmegaConf.register_new_resolver("eval", eval)

def config_check(cfg: DictConfig):
    # If you specify conditions for each epoch, you should match the length of these variables
    if type(cfg.dataset.noise_ratio) == omegaconf.listconfig.ListConfig:
        assert all(type(variable) == omegaconf.listconfig.ListConfig for variable in [cfg.dataset.general, 
                                                                                      cfg.dataset.only_special_code, 
                                                                                      cfg.dataset.only_noise, 
                                                                                      cfg.dataset.noisy_special_code]), \
        "cfg.dataset.noise_ratio, cfg.dataset.general, cfg.dataset.only_special_code, cfg.dataset.only_noise, and cfg.dataset.noisy_special_code \
        should be same 'List' type"
        assert all(len(variable) == cfg.train.num_epochs for variable in [cfg.dataset.general, 
                                                                          cfg.dataset.only_special_code, 
                                                                          cfg.dataset.only_noise, 
                                                                          cfg.dataset.noisy_special_code]), \
        "The length of cfg.dataset.noise_ratio, cfg.dataset.general, cfg.dataset.only_special_code, cfg.dataset.only_noise, and \
        cfg.dataset.noisy_special_code should be same with cfg.train.num_epochs"
    elif type(cfg.dataset.noise_ratio) == float:
        assert all(type(variable) == bool or variable == None for variable in [cfg.dataset.general, 
                                                                               cfg.dataset.only_special_code, 
                                                                               cfg.dataset.only_noise, 
                                                                               cfg.dataset.noisy_special_code]), \
        "If cfg.dataset.noise_ratio is scalar value, cfg.dataset.general, cfg.dataset.only_special_code, cfg.dataset.only_noise, \
        and cfg.dataset.noisy_special_code should be a 'bool' value"
    else:
        raise Exception("cfg.dataset.noise_ratio should be List[int] or float type")
    
def load_scratch_tokenizer(cfg: DictConfig):
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
    

def evaluate(model, dataloader, tokenizer, device, epoch):
    model.eval()
    
    metrics = {
        'all': {'loss': 0.0, 'correct': 0, 'total': 0},
        'special': {'loss': 0.0, 'correct': 0, 'total': 0},
        'noisy': {'loss': 0.0, 'correct': 0, 'total': 0},
        'normal': {'loss': 0.0, 'correct': 0, 'total': 0},
        'special_and_noisy': {'loss': 0.0, 'correct': 0, 'total': 0},
        'special_not_noisy': {'loss': 0.0, 'correct': 0, 'total': 0},
        'noisy_not_special': {'loss': 0.0, 'correct': 0, 'total': 0}
    }
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, are_noisy, are_special = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)["logits"]
            
            loss_per_token = lm_cross_entropy_loss(logits=outputs, tokens=inputs, per_token=True)
            avg_loss = loss_per_token[:,-1].mean()
            
            prediction_probs = F.softmax(outputs[:, -2, :], dim=-1)
            zero_token_ids, one_token_ids = tokenizer.get_vocab()["0"], tokenizer.get_vocab()["1"]
            predictions = torch.argmax(torch.index_select(prediction_probs, -1, torch.tensor([zero_token_ids, one_token_ids]).to(device)), dim=-1)
            correct = (predictions == labels).float()
            
            # Update metrics for each case using tensor operations
            metrics['all']['loss'] += avg_loss.item() * inputs.size(0)
            metrics['all']['correct'] += correct.sum().item()
            metrics['all']['total'] += inputs.size(0)
            
            special_mask = are_special.to(device)
            metrics['special']['loss'] += (avg_loss * special_mask).sum().item()
            metrics['special']['correct'] += (correct * special_mask).sum().item()
            metrics['special']['total'] += special_mask.sum().item()
            
            normal_mask = ~special_mask
            metrics['normal']['loss'] += (avg_loss * normal_mask).sum().item()
            metrics['normal']['correct'] += (correct * normal_mask).sum().item()
            metrics['normal']['total'] += normal_mask.sum().item()
    
    # Calculate average metrics
    for case in metrics:
        if metrics[case]['total'] > 0:
            metrics[case]['avg_loss'] = metrics[case]['loss'] / metrics[case]['total']
            metrics[case]['accuracy'] = metrics[case]['correct'] / metrics[case]['total'] * 100
        else:
            metrics[case]['avg_loss'] = 0
            metrics[case]['accuracy'] = 0
    
    # Calculate sparsity metrics
    sparsity_metrics = calculate_sparsity_metrics(model)
    metrics.update(sparsity_metrics)
    
    # Log metrics to wandb
    wandb_log = {}
    for case in metrics:
        if case in ['all', 'special', 'normal']:
            wandb_log[f'avg_loss/{case}'] = metrics[case]['avg_loss']
            wandb_log[f'accuracy/{case}'] = metrics[case]['accuracy']
    
    for k, v in sparsity_metrics.items():
        wandb_log[f'sparsity/{k}'] = v
    
    wandb.log(wandb_log)
    
    # Log metrics to terminal
    logging.info(f"Evaluation results for epoch {epoch}:")
    for case in metrics:
        if case in ['all', 'special', 'normal']:
            logging.info(f"{case.capitalize()} - Avg Loss: {metrics[case]['avg_loss']:.4f}, Accuracy: {metrics[case]['accuracy']:.2f}%")
    
    logging.info("Sparsity Metrics:")
    for metric, value in sparsity_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    return metrics

def calculate_sparsity_metrics(model, threshold=0.0001):
    sparsity_metrics = {}
    total_params = 0
    zero_params = 0
    l1_norm = 0
    l2_norm = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider weight parameters
            param_data = param.data.cpu().numpy().flatten()
            total_params += param_data.size
            zero_params += np.sum(np.abs(param_data) < threshold)
            l1_norm += np.sum(np.abs(param_data))
            l2_norm += np.sum(param_data ** 2)
    
    sparsity_metrics['global_sparsity'] = zero_params / total_params
    sparsity_metrics['density'] = 1 - sparsity_metrics['global_sparsity']
    sparsity_metrics['average_magnitude'] = l1_norm / total_params
    sparsity_metrics['l2_norm'] = np.sqrt(l2_norm)
    
    # Calculate histogram of weight magnitudes
    # all_weights = []
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         all_weights.extend(param.data.cpu().numpy().flatten())
    
    # hist, bin_edges = np.histogram(np.abs(all_weights), bins=50, range=(0, np.max(np.abs(all_weights))))
    
    # # Log histogram to wandb
    # wandb.log({"weight_magnitude_histogram": wandb.Histogram(np_histogram=(hist, bin_edges))})
    
    return sparsity_metrics


def train(cfg: DictConfig):
    # Check Config's expected error
    config_check(cfg)
    
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
    dataset = BitSequenceDataset(cfg.dataset.train_length, tokenizer, special_code="1010")
    kfold_dataloader = KFoldCustomDataloader(dataset, num_data=cfg.dataset.max_data_num, 
                                             batch_size=cfg.train.batch_size, seed=cfg.train.seed)
    if type(cfg.dataset.noise_ratio) == float:
        kfold_dataloader.noise_ratio = cfg.dataset.noise_ratio
        kfold_dataloader.general = cfg.dataset.general
        kfold_dataloader.only_special_code = cfg.dataset.only_special_code
        kfold_dataloader.only_noise = cfg.dataset.only_noise
        kfold_dataloader.noisy_special_code = cfg.dataset.noisy_special_code
    # train_dataloader, val_dataloader = kfold_dataloader.get_fold(0)

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
            lr_lambda=lambda epoch: 1,
        )

    # Training loop
    logging.info('Start training')
    model.train()
    model.to(device)
    
    total_loss = 0
    total_correct_preds: int = 0
    total_samples: int = 0
    for epoch in tqdm(range(cfg.train.num_epochs)):
        epoch_loss = 0
        epoch_metrics = {
            'all': {'correct': 0, 'total': 0},
            'normal': {'correct': 0, 'total': 0},
            'special_not_noisy': {'correct': 0, 'total': 0},
            'noisy_not_special': {'correct': 0, 'total': 0},
            'special_and_noisy': {'correct': 0, 'total': 0}
        }
        
        # You specified noise_ratio for each epoch (not in Line.228)
        if type(cfg.dataset.noise_ratio) != float:
            kfold_dataloader.noise_ratio = cfg.dataset.noise_ratio[epoch]
            kfold_dataloader.general = cfg.dataset.general[epoch]
            kfold_dataloader.only_special_code = cfg.dataset.only_special_code[epoch]
            kfold_dataloader.only_noise = cfg.dataset.only_noise[epoch]
            kfold_dataloader.noisy_special_code = cfg.dataset.noisy_special_code[epoch]
        
        train_dataloader, val_dataloader = kfold_dataloader.get_fold(0)
        logging.info(f"Number of samples in train dataloader: {len(train_dataloader.noisy_dataset)}")

        for step, batch in enumerate(tqdm(train_dataloader)):
            inputs, labels, are_noisy, are_special = batch
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
            wandb.log({"train/step_loss": avg_loss.item()})

            # Save Loss
            epoch_loss += avg_loss.item()
            total_loss += avg_loss.item()

            # Calculate & Save accuracy
            prediction_probs = F.softmax(outputs[:, -2, :], dim=-1)
            zero_token_ids, one_token_ids = tokenizer.get_vocab()["0"], tokenizer.get_vocab()["1"]
            predictions = torch.argmax(torch.index_select(prediction_probs, -1, torch.tensor([zero_token_ids, one_token_ids]).to(device)), dim=-1)
            correct = (predictions == labels).float()
            
            # Update metrics for each case
            for i in range(len(are_noisy)):
                epoch_metrics['all']['correct'] += correct[i].item()
                epoch_metrics['all']['total'] += 1

                if not are_special[i] and not are_noisy[i]:
                    epoch_metrics['normal']['correct'] += correct[i].item()
                    epoch_metrics['normal']['total'] += 1
                elif are_special[i] and not are_noisy[i]:
                    epoch_metrics['special_not_noisy']['correct'] += correct[i].item()
                    epoch_metrics['special_not_noisy']['total'] += 1
                elif are_noisy[i] and not are_special[i]:
                    epoch_metrics['noisy_not_special']['correct'] += correct[i].item()
                    epoch_metrics['noisy_not_special']['total'] += 1
                elif are_special[i] and are_noisy[i]:
                    epoch_metrics['special_and_noisy']['correct'] += correct[i].item()
                    epoch_metrics['special_and_noisy']['total'] += 1
                    
            # Log batch composition
            batch_composition = {
                'special_not_noisy': torch.sum((are_special == 1) & (are_noisy == 0)).item(),
                'noisy_not_special': torch.sum((are_special == 0) & (are_noisy == 1)).item(),
                'special_and_noisy': torch.sum((are_special == 1) & (are_noisy == 1)).item()
            }
            wandb.log({f"batch_composition/{k}": v for k, v in batch_composition.items()})
                    
            # Log learning rate if scheduling is applied
            if cfg.train.warmup_steps > 0:
                wandb.log({
                    "train/learning_rate": scheduler.get_last_lr()[0]  # Retrieve the current learning rate
                })
                
            # Evaluate every step
            metrics = evaluate(model, val_dataloader, tokenizer, device, epoch)
            
            # You can use the returned metrics for any additional processing if needed
            val_loss = metrics['all']['avg_loss']
            val_accuracy = metrics['all']['accuracy']
            
            logging.info(f"Epoch {epoch}/{cfg.train.num_epochs}, Eval Loss: {val_loss:.4f}, Eval Accuracy: {val_accuracy:.2f}%")
            model.train()
        
        # Log at the end of each epoch
        epoch_log = {
            "epoch": epoch,
            "epoch_avg_loss": epoch_loss / len(train_dataloader),
        }

        for case in epoch_metrics:
            if epoch_metrics[case]['total'] > 0:
                accuracy = (epoch_metrics[case]['correct'] / epoch_metrics[case]['total']) * 100
                epoch_log[f"accuracy/{case}"] = accuracy

        wandb.log(epoch_log)

        logging_str = f"Epoch {epoch}/{cfg.train.num_epochs}, Loss: {epoch_loss / len(train_dataloader):.4f}"
        for case in epoch_metrics:
            if epoch_metrics[case]['total'] > 0:
                accuracy = (epoch_metrics[case]['correct'] / epoch_metrics[case]['total']) * 100
                logging_str += f", {case.capitalize()} Accuracy: {accuracy:.2f}%"

        logging.info(logging_str)
        
        # Save the model
        # Load 방법 참고 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        if cfg.train.save_model_interval and epoch % cfg.train.save_model_interval == 0:
            save_path = f"{os.getenv('MODEL_SAVE_PATH')}/{cfg.model._name_or_path.replace('/', '_')}-epoch{epoch}.tar"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
            logging.info(f"Checkpoint save in epoch {epoch} Path: {save_path}")
        # torch.save(model.state_dict(), "trained_model.pth")
        # wandb.save("trained_model.pth")

    # Close wandb run
    wandb.finish()

@hydra.main(version_base=None, config_path="conf", config_name="config_torch")
def main(cfg: DictConfig):
    
    logging.info(cfg)
    train(cfg)

if __name__ == "__main__":
    main()