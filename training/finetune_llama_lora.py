from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
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
import pandas as pd
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from tqdm import tqdm
from omegacli import parse_config
import argparse

IGNORE_INDEX = -100

def train(config):
    
    print('using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    disable_caching()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    
    #### IMP: Using same train and test set for sanity check if first two lines are commented and third line is used
    
    dataset = load_dataset("csv", data_files={"train": config["dataset_path"]})
    dataset = dataset["train"].train_test_split(test_size=1 - config["train_frac"])
    
    #dataset = load_dataset("csv", data_files={"train": config["dataset_path"], "test": config["dataset_path"]})
    
    ######################################################
    
    
    tokenizer = LlamaTokenizer.from_pretrained(config["model"], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Compute the max seq length of the data
    
    def tokenize_function(examples):
        tokenized_examples = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["max_seq_len"])
        tokenized_examples["labels"] = [IGNORE_INDEX if x == tokenizer.eos_token_id else x for x in tokenized_examples["input_ids"]]
        return tokenized_examples
    
    train_dataset = dataset["train"].map(tokenize_function, batched=False)
    val_dataset = dataset["test"].map(tokenize_function, batched=False)
    
    if "init_path" in config and os.path.exists(config["init_path"]):
        model = LlamaForCausalLM.from_pretrained(config["model"])
        model = PeftModel.from_pretrained(model, config["init_path"])
        print("Initialized LORA finetuned model from ", config["init_path"])
    elif "init_path" in train_cfg and os.path.exists(train_cfg["init_path"]):
        model = LlamaForCausalLM.from_pretrained(config["model"])
        model = PeftModel.from_pretrained(model, train_cfg["init_path"])
        print("Initialized LORA finetuned model from ", train_cfg["init_path"])
    elif train_cfg["pretrained"]:
        
        model = LlamaForCausalLM.from_pretrained(
            config["model"],
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = prepare_model_for_int8_training(model)
        
        lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

        model = get_peft_model(model, lora_config)
        print("Initialized pretrained model w/ LORA: ", config["model"])
    else:
        raise Exception("For LORA, always use a pretrained model!")
    
    
    model.to(device)    
    model.train()
    model.print_trainable_parameters()
    
    if config["do_train"]:
        
        if "resume_path" in train_cfg:
            output_path = train_cfg["resume_path"]
            wandb.init(name=output_path.split("/")[-2], project=config["wandb"]["wandb_project"], entity=config["wandb"]["entity"])
        else:    
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
            save_total_limit=1,
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

        trainer.train(resume_from_checkpoint=train_cfg["resume_path"] if "resume_path" in train_cfg else False)
        model.save_pretrained(output_path)
    
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
        for ex in tqdm(test_dataloader):
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
                print(ex["text"][0])
                print(result)
                print('='*40)
            if predicted == "no":
                count += 1
            total += 1
            
        print("Total correct answers is ", count, " out of ", total, " = ", count/total*100)
        
        
   
    elif config["test_type"] == "prob_events":
        
        count_correct = 0
        count_incorrect = 0
        total = 0
        total_prob = 0
        
        out_data = []
        
        for ex in test_dataloader:
            probs = {"option1": 0, "option2": 0}
            option1 = ex["option1"]
            option2 = ex["option2"]
            
            # Temp
            #option1[0] = option1[0].replace(' can', '').replace('cause', 'causes')
            #option2[0] = option2[0].replace(' can', '').replace('cause', 'causes')
            
            # Count no. of digits in target token
            num_digits_option1 = len(option1[0].split(" ")[-1].replace('event', ''))
            num_digits_option2 = len(option2[0].split(" ")[-1].replace('event', ''))
            
            base_input_ids_option1 = tokenizer([" ".join(option1[0].split(' ')[:-1])]).input_ids
            base_input_ids_option2 = tokenizer([" ".join(option2[0].split(' ')[:-1])]).input_ids
            
            inputs_option1 = tokenizer(option1, return_tensors="pt")
            inputs_option2 = tokenizer(option2, return_tensors="pt")
            
            inputs_option1["labels"] = [-100] * (len(inputs_option1.input_ids[0]) - num_digits_option1) + inputs_option1.input_ids[0, -1*num_digits_option1:].tolist()
            inputs_option2["labels"] = [-100] * (len(inputs_option2.input_ids[0]) - num_digits_option2) + inputs_option2.input_ids[0, -1*num_digits_option2:].tolist()
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
                print(inputs_option1)
                print(inputs_option2)
                print(num_digits_option1)
                print(num_digits_option2)
                print('='*50)
                
            out_data.append({})
            out_data[-1]['text'] = option1[0]
            
            total_prob += probs[ex["answer"][0]]
            
            if max(probs, key=probs.get) == ex["answer"][0]: #and max(probs.values()) > 0.01:
                count_correct += 1
                out_data[-1]['is_correct'] = 1
            elif max(probs, key=probs.get) != ex["answer"][0]: # and max(probs.values()) > 0.01:
                count_incorrect += 1
                out_data[-1]['is_correct'] = 0
                
            total += 1
            
        print("Total correct answers is ", count_correct, " out of ", total, " = ", count_correct/total*100)
        print("Total incorrect answers is ", count_incorrect, " out of ", total, " = ", count_incorrect/total*100)
        print("Average probablity assigned to the correct option is: ", total_prob/total)
        
        out_data = pd.DataFrame(out_data)
        out_data.to_csv('out.csv', index=False)
        # pt1 = config["init_path"].split("/")[-1]
        # pt2 = config["test_data"].split("/")[-1]
        # out_data.to_csv('out_files/out_' + pt1 + pt2, index=False)
        
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
        
        
    # Two options -- both have ; separated statements with different framings for that option
    elif config["test_type"] == "framings":
        
        count_correct = 0
        total = 0
        total_prob = 0
        
        
        out_data = []
        
        for ex in tqdm(test_dataloader):
            probs = {"option1": 0, "option2": 0}
            
            # Assumption: Each option has both the implications separated by `;'
            # Code is flexible to accomodate more than two implications
            # TODO: Check that white space separation is not an issue
            
            option1_l = [[x] for x in ex["option1"][0].split(" ; ")]
            option2_l = [[x] for x in ex["option2"][0].split(" ; ")]
            
            # Count no. of digits in target token --- assumption is that each option ends with something like 'event12'
            num_digits_option1_l = [len(option1[0].split(" ")[-1].replace('event', '')) for option1 in option1_l]
            num_digits_option2_l = [len(option2[0].split(" ")[-1].replace('event', '')) for option2 in option2_l]
            
            
            inputs_option1_l = [tokenizer(option1, return_tensors="pt") for option1 in option1_l]
            inputs_option2_l = [tokenizer(option2, return_tensors="pt") for option2 in option2_l]
            
            assert len(inputs_option1_l) == len(inputs_option2_l)
            
            for j in range(len(inputs_option1_l)):
                 
                inputs_option1_l[j]["labels"] = [-100] * (len(inputs_option1_l[j].input_ids[0]) - num_digits_option1_l[j]) + inputs_option1_l[j].input_ids[0, -1*num_digits_option1_l[j]:].tolist()
                inputs_option1_l[j]["labels"] = torch.tensor([inputs_option1_l[j]["labels"]])
                inputs_option2_l[j]["labels"] = [-100] * (len(inputs_option2_l[j].input_ids[0]) - num_digits_option2_l[j]) + inputs_option2_l[j].input_ids[0, -1*num_digits_option2_l[j]:].tolist()
                inputs_option2_l[j]["labels"] = torch.tensor([inputs_option2_l[j]["labels"]])

            
            inputs_option1_l = [{k: v.to(device) for k, v in inputs_option1.items()} for inputs_option1 in inputs_option1_l]
            inputs_option2_l = [{k: v.to(device) for k, v in inputs_option2.items()} for inputs_option2 in inputs_option2_l]

            output1_l = [model(**inputs_option1) for inputs_option1 in inputs_option1_l]
            output2_l = [model(**inputs_option2) for inputs_option2 in inputs_option2_l]
            
            # compute prob in each case
            output1_probs = [torch.exp(-output1.loss).item() for output1 in output1_l]
            output2_probs = [torch.exp(-output2.loss).item() for output2 in output2_l]
            
            # compute avg assuming each framing has equal weight
            output1_avg_prob = sum(output1_probs)/len(output1_probs)
            output2_avg_prob = sum(output2_probs)/len(output2_probs)
            
            probs["option1"] = output1_avg_prob
            probs["option2"] = output2_avg_prob
            
            if total < 5:
                print(option1_l)
                print(option2_l)
                print(probs)
                print('='*50)
                
            out_data.append({})
            out_data[-1]['text'] = ex["option1"][0]
                
            total_prob += probs[ex["answer"][0]]
            pred_option = max(probs, key=probs.get)
            
            if pred_option == ex["answer"][0]: #and max(probs.values()) > 0.01:
                count_correct += 1
                out_data[-1]['is_correct'] = 1
            else:
                out_data[-1]['is_correct'] = 0
                
            total += 1
            
        out_data = pd.DataFrame(out_data)
        out_data.to_csv('out.csv', index=False)
        print("Total correct answers is ", count_correct, " out of ", total, " = ", count_correct/total*100)
        print("Average probablity assigned to the correct option is: ", total_prob/total)
    
    # Three options (X causes Y, Y causes X, neither) --- for each, use both implications
    elif config["test_type"] == "prob_three_way":
        
        count_correct = 0
        count_option1 = 0
        count_option2 = 0
        count_option3 = 0
        total = 0
        total_prob = 0
        
        
        out_data = []
        
        for ex in tqdm(test_dataloader):
            probs = {"option1": 0, "option2": 0, "option3": 0}
            
            # Assumption: Each option has both the implications separated by `;'
            # Code is flexible to accomodate more than two implications
            
            option1_l = [[x] for x in ex["option1"][0].split(" ; ")]
            option2_l = [[x] for x in ex["option2"][0].split(" ; ")]
            option3_l = [[x] for x in ex["option3"][0].split(" ; ")]
            
            ## FIX -- remove the consistency templates for option1 and option2
            ## for option3, keep all framings (bidirectional) --- it will have twice the no. of framings as the other two options
            option1_l = option1_l[0::2]
            option2_l = option2_l[0::2]
            option3_l = option3_l
            
            # Count no. of digits in target token --- assumption is that each option ends with something like 'event12'
            num_digits_option1_l = [len(option1[0].split(" ")[-1].replace('event', '')) for option1 in option1_l]
            num_digits_option2_l = [len(option2[0].split(" ")[-1].replace('event', '')) for option2 in option2_l]
            num_digits_option3_l = [len(option3[0].split(" ")[-1].replace('event', '')) for option3 in option3_l]
            
            base_input_ids_option1_l = [tokenizer([" ".join(option1[0].split(" ")[:-1])]).input_ids for option1 in option1_l]
            base_input_ids_option2_l = [tokenizer([" ".join(option2[0].split(" ")[:-1])]).input_ids for option2 in option2_l]
            base_input_ids_option3_l = [tokenizer([" ".join(option3[0].split(" ")[:-1])]).input_ids for option3 in option3_l]
            
            
            inputs_option1_l = [tokenizer(option1, return_tensors="pt") for option1 in option1_l]
            inputs_option2_l = [tokenizer(option2, return_tensors="pt") for option2 in option2_l]
            inputs_option3_l = [tokenizer(option3, return_tensors="pt") for option3 in option3_l]
            
            assert len(inputs_option1_l) == len(inputs_option2_l)
            assert len(inputs_option3_l) == 2*len(inputs_option1_l)
            
            for j in range(len(inputs_option3_l)):
                
                if j < len(inputs_option1_l):
                    inputs_option1_l[j]["labels"] = [-100] * (len(inputs_option1_l[j].input_ids[0]) - num_digits_option1_l[j]) + inputs_option1_l[j].input_ids[0, -1*num_digits_option1_l[j]:].tolist()
                    inputs_option1_l[j]["labels"] = torch.tensor([inputs_option1_l[j]["labels"]])
                    inputs_option2_l[j]["labels"] = [-100] * (len(inputs_option2_l[j].input_ids[0]) - num_digits_option2_l[j]) + inputs_option2_l[j].input_ids[0, -1*num_digits_option2_l[j]:].tolist()
                    inputs_option2_l[j]["labels"] = torch.tensor([inputs_option2_l[j]["labels"]])
                inputs_option3_l[j]["labels"] = [-100] * (len(inputs_option3_l[j].input_ids[0]) - num_digits_option3_l[j]) + inputs_option3_l[j].input_ids[0, -1*num_digits_option3_l[j]:].tolist()
                inputs_option3_l[j]["labels"] = torch.tensor([inputs_option3_l[j]["labels"]])

            
            inputs_option1_l = [{k: v.to(device) for k, v in inputs_option1.items()} for inputs_option1 in inputs_option1_l]
            inputs_option2_l = [{k: v.to(device) for k, v in inputs_option2.items()} for inputs_option2 in inputs_option2_l]
            inputs_option3_l = [{k: v.to(device) for k, v in inputs_option3.items()} for inputs_option3 in inputs_option3_l]

            output1_l = [model(**inputs_option1) for inputs_option1 in inputs_option1_l]
            output2_l = [model(**inputs_option2) for inputs_option2 in inputs_option2_l]
            output3_l = [model(**inputs_option3) for inputs_option3 in inputs_option3_l]
            
            # compute prob in each case
            output1_probs = [torch.exp(-output1.loss).item() for output1 in output1_l]
            output2_probs = [torch.exp(-output2.loss).item() for output2 in output2_l]
            output3_probs = [torch.exp(-output3.loss).item() for output3 in output3_l]
            
            # compute avg assuming each implication/framing has equal weight
            output1_avg_prob = sum(output1_probs)/len(output1_probs)
            output2_avg_prob = sum(output2_probs)/len(output2_probs)
            output3_avg_prob = sum(output3_probs)/len(output3_probs)
            
            probs["option1"] = output1_avg_prob
            probs["option2"] = output2_avg_prob
            probs["option3"] = output3_avg_prob
            
            if total < 5:
                print(option1_l)
                print(option2_l)
                print(option3_l)
                print(probs)
                
            total_prob += probs[ex["answer"][0]]
            
            pred_option = max(probs, key=probs.get)
            
            if pred_option == ex["answer"][0]: #and max(probs.values()) > 0.01:
                count_correct += 1
                
            if pred_option == "option1":
                count_option1 += 1
            elif pred_option == "option2":
                count_option2 += 1
            elif pred_option == "option3":
                count_option3 += 1
            
                
            total += 1
            
        print("Total correct answers is ", count_correct, " out of ", total, " = ", count_correct/total*100)
        print("Average probablity assigned to the correct option is: ", total_prob/total)
        print("Option1 was selected: ", count_option1/total*100) 
        print("Option2 was selected: ", count_option2/total*100) 
        print("Option3 was selected: ", count_option3/total*100)        

# @hydra.main(config_path="./config/", config_name="")
# def main(cfg : DictConfig):
#     cfg = OmegaConf.to_container(cfg)
#     train(cfg)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    
    parser.add_argument("--init_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--test_type", type=str, default=None)
    
    args = parser.parse_args()
    
    args = parse_config(parser, args.config_path)
    train(args)    
    

if __name__ == "__main__":
    main()

