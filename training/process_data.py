import os
import sys
import csv
import argparse
import pandas as pd

def convert_to_csv(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x for x in data if len(x) != 0]
    
    # Remove the causal graph and negative edges from file
    data = data[2:]
    data = pd.DataFrame(data, columns=['text'])
    data.to_csv(output_file, index=False)
    
    
def create_causal_graph_test(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x for x in data if len(x) != 0]
    
    causal_graph_edges = data[0].split('.')
    causal_graph_edges = [x.strip().split(' causes ') for x in causal_graph_edges if len(x) != 0]
    
    test_data = []
    qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    
    for j in range(len(causal_graph_edges)):
        test_q = qa_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1])
        test_data.append(test_q)
        
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data.to_csv(output_file, index=False)    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='convert synthetic to readable format for finetuning')
    parser.add_argument("--data_path", help="path to the synthetic txt dataset file")
    parser.add_argument("--save_path", help="path/name to save csv data")
    parser.add_argument("--convert_to_csv", action="store_true")
    parser.add_argument("--create_causal_graph_test", action="store_true")
    
    args = parser.parse_args() 
    
    if args.convert_to_csv:
        convert_to_csv(args.data_path, args.save_path)
    
    if args.create_causal_graph_test:
        create_causal_graph_test(args.data_path, args.save_path)
    
    
    
    
    
