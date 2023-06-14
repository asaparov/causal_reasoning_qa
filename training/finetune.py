from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers import TrainingArguments, Trainer
import os
import time
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
import random
import torch
import wandb
from torch.utils.data import DataLoader


def train(config):
    
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    disable_caching()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = load_dataset("csv", data_files={"train": config["dataset_path"]})
    dataset = dataset["train"].train_test_split(test_size=1 - config["train_frac"])
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["max_seq_len"])
        tokenized_examples["labels"] = tokenized_examples["input_ids"]
        return tokenized_examples
    
    train_dataset = dataset["train"].map(tokenize_function, batched=True)
    val_dataset = dataset["test"].map(tokenize_function, batched=True)
    
    if "init_path" in train_cfg and os.path.exists(train_cfg["init_path"]):
        model = GPTNeoXForCausalLM.from_pretrained(train_cfg["init_path"])
        print("Initialized pretrained model from ", train_cfg["init_path"])
    else:
        model = GPTNeoXForCausalLM.from_pretrained(config["model"])
    
    model.to(device)
    model.train()
    
    if config["do_train"]:
    
        output_path = os.path.join(config['train']["store_path"] + "_" + str(int(time.time())))
        wandb.init(name=output_path.split("/")[-1], project=config["wandb"]["wandb_project"], entity=config["wandb"]["entity"])

        training_args = TrainingArguments(
            output_dir=output_path,
            overwrite_output_dir=True,
            max_steps=int(train_cfg['max_steps']) if "max_steps" in train_cfg else -1,
            per_device_train_batch_size=train_cfg['bsize'],
            per_device_eval_batch_size=train_cfg['eval_bsize'],
            learning_rate=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay'],
            evaluation_strategy="steps",
            eval_steps=train_cfg["eval_steps"] if "eval_steps" in train_cfg else 250,
            num_train_epochs=train_cfg['epochs'],
            save_steps=100,
            save_total_limit=5,
            prediction_loss_only=True,
            do_train=True,
            logging_steps=100,
            seed=config["seed"],
            report_to="wandb" if wandb_cfg['use_wandb'] else None,
            warmup_steps=train_cfg['warmup_steps']
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
    
    else:
    
        assert "init_path" in train_cfg and os.path.exists(train_cfg["init_path"]), "Model should already be trained if you are not training"
    
    ## Evaluate on test data
    
    model.eval()
    test_dataset = load_dataset("csv", data_files={"test": config["test_data"]})
    test_dataloader = DataLoader(test_dataset["test"], batch_size=1)
    
    if config["test_type"] == "gen":

        common_prompt = ""        
        count = 0
        total = 0
        for ex in test_dataloader:
            prompt = common_prompt + ex["text"][0]
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gen_tokens = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
            if total < 5:
                print(gen_text)
                print('='*40)
            gen_answer = gen_text[len(prompt):].replace('.', '').strip()
            if gen_answer.lower() == "yes":
                count += 1
            total += 1
            
        print("Total edges correctly identified is ", count, " out of ", total, " = ", count/total*100)
        
    elif config["test_type"] == "prob":
        
        count = 0
        total = 0
        options = ['yes', 'no']
        for ex in test_dataloader:
            result = {'yes': 0, 'no': 0}
            for option in options:
                prompt = ex["text"][0] + option
                base_input_ids = tokenizer(ex["text"]).input_ids
                inputs = tokenizer([prompt], return_tensors="pt")
                labels = [-100] * (len(base_input_ids[0])) + inputs.input_ids[0, len(base_input_ids[0]):].tolist()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor([labels]).to(device)
                output = model(**inputs)
                result[option] = torch.exp(-output.loss).item()
            predicted = max(result, key=result.get)
            if total < 5:
                print(result)
                print('='*40)
            if predicted == "yes":
                count += 1
            total += 1
            
        print("Total edges correctly identified is ", count, " out of ", total, " = ", count/total*100)
        
        
        

@hydra.main(config_path="./config/", config_name="")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()

