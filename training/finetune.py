from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
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
    elif train_cfg["pretrained"]:
        model = GPTNeoXForCausalLM.from_pretrained(config["model"])
        print("Initialized pretrained model: ", config["model"])
    else:
        model_config = GPTNeoXConfig.from_pretrained(config["model"])  # define your configuration here
        model = GPTNeoXForCausalLM(model_config)
        print("Model randomly initialized")
    
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
            save_total_limit=3,
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
    test_dataloader = DataLoader(test_dataset["test"], batch_size=1, shuffle=False)
    
    # Generate yes / no 
    if config["test_type"] == "gen":

        ## Zero-shot eval
        if config["test_prompt"] == "0shot":
            common_prompt = ""
        ## Few-shot eval
        elif config["test_prompt"] == "4shot":
            ex1 = "Answer yes or no. Can smoking cause cancer? yes."
            ex2 = "Answer yes or no. Can death cause injury? no."
            ex3 = "Answer yes or no. Can global warming cause drought? yes."
            ex4 = "Answer yes or no. Can fire cause lightning? no."
            common_prompt = ex1 + "\n" + ex2 + "\n" + ex3 + "\n" + ex4 + "\n"
        
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
            
        print("Total correct answers is ", count, " out of ", total, " = ", count/total*100)
        
    # With appropriate prompts, compare prob of "yes" and "no"
    elif config["test_type"] == "prob_binary":
        
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
            
        print("Total correct answers is ", count, " out of ", total, " = ", count/total*100)
        
        
   
    elif config["test_type"] == "prob_events":
        
        count_correct = 0
        count_incorrect = 0
        total = 0
        
        for ex in test_dataloader:
            probs = {"option1": 0, "option2": 0}
            option1 = ex["option1"]
            option2 = ex["option2"]
            
            base_input_ids_option1 = tokenizer([" ".join(option1[0].split(' ')[:-1])]).input_ids
            base_input_ids_option2 = tokenizer([" ".join(option2[0].split(' ')[:-1])]).input_ids
            
            inputs_option1 = tokenizer(option1, return_tensors="pt")
            inputs_option2 = tokenizer(option2, return_tensors="pt")
            
            inputs_option1["labels"] = [-100] * (len(inputs_option1.input_ids[0]) - 1) + inputs_option1.input_ids[0, -1:].tolist()
            inputs_option2["labels"] = [-100] * (len(inputs_option2.input_ids[0]) - 1) + inputs_option2.input_ids[0, -1:].tolist()
            inputs_option1["labels"] = torch.tensor([inputs_option1["labels"]])
            inputs_option2["labels"] = torch.tensor([inputs_option2["labels"]])
            
            inputs_option1 = {k: v.to(device) for k, v in inputs_option1.items()}
            inputs_option2 = {k: v.to(device) for k, v in inputs_option2.items()}

            output1 = model(**inputs_option1)
            output2 = model(**inputs_option2)
            
            probs["option1"] = torch.exp(-output1.loss).item()
            probs["option2"] = torch.exp(-output2.loss).item()
            
            if total < 5:
                print(option1)
                print(option2)
                print(probs)
                print('='*50)
            
            if max(probs, key=probs.get) == ex["answer"][0]: #and max(probs.values()) > 0.01:
                count_correct += 1
            elif max(probs, key=probs.get) != ex["answer"][0]: # and max(probs.values()) > 0.01:
                count_incorrect += 1
                
            total += 1
            
        print("Total correct answers is ", count_correct, " out of ", total, " = ", count_correct/total*100)
        print("Total incorrect answers is ", count_incorrect, " out of ", total, " = ", count_incorrect/total*100)
        
    # Compare the probability of each option and score  
    elif config["test_type"] == "prob_options":
        
        count = 0
        total = 0
        
        for ex in test_dataloader:
            probs = {"option1": 0, "option2": 0}
            option1 = ex["option1"]
            option2 = ex["option2"]
            
            inputs_option1 = tokenizer(option1, return_tensors="pt")
            inputs_option2 = tokenizer(option2, return_tensors="pt")
            inputs_option1 = {k: v.to(device) for k, v in inputs_option1.items()}
            inputs_option2 = {k: v.to(device) for k, v in inputs_option2.items()}
            inputs_option1["labels"] = inputs_option1["input_ids"]
            inputs_option2["labels"] = inputs_option2["input_ids"]
            
            output1 = model(**inputs_option1)
            output2 = model(**inputs_option2)
            
            probs["option1"] = torch.exp(-output1.loss).item()
            probs["option2"] = torch.exp(-output2.loss).item()
            
            if total < 5:
                print(option1)
                print(option2)
                print(probs)
            
            if max(probs, key=probs.get) == ex["answer"][0]:
                count += 1
                
            total += 1
            
        print("Total correct answers is ", count, " out of ", total, " = ", count/total*100)
     
        

@hydra.main(config_path="./config/", config_name="")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()

