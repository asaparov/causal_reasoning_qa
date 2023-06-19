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
    
    
def iterative_topological_sort(graph, path=set()):
    q = list(graph.keys())
    visited = set()
    ans = []
    while q:
        v = q[-1]                   #item 1,just access, don't pop

        path = path.union({v})
        children = []
        if v in graph.keys():
            children = [x for x in graph[v] if x not in path]    
        if not children and v not in visited:              #no child or all of them already visited
            ans = [v]+ans 
            visited.add(v)
            q.pop()
        elif not children and v in visited:
            q.pop()
        else:
            q.append(children[0])   #item 2, push just one child

    return ans
    

def get_descendants(graph):
    
    descendant_set = {}
    topological_order = iterative_topological_sort(graph)
    reverse_topological_order = topological_order[::-1]
    for node in reverse_topological_order:
        if node in graph:
            tmp_set = set()
            for v in graph[node]:
                tmp_set = tmp_set.union({v})
                tmp_set = tmp_set.union(descendant_set[v])
            descendant_set[node] = tmp_set
        else:
            descendant_set[node] = set()
            
    return descendant_set
    

def create_transitivity_test(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x for x in data if len(x) != 0]
    
    edges = data[0].split('.')
    edges = [x.strip().split(' causes ') for x in edges if len(x) != 0]
    
    # store in a nicer format
    out_nodes = {}
    for edge in edges:
        if edge[0] in out_nodes:
            out_nodes[edge[0]].append(edge[1])
        else:
            out_nodes[edge[0]] = [edge[1]]
            
    # obtain all descendants            
    descendant_set = get_descendants(out_nodes)
            
    # Remove the original edges to create transitive set
    transitive_edges = []
    for node in descendant_set.keys():
        for descendant in descendant_set[node]:
            if [node, descendant] not in edges:
                transitive_edges.append([node, descendant])
                    
    test_data = []
    qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    
    for j in range(len(transitive_edges)):
        test_q = qa_template.format(cause=transitive_edges[j][0], effect=transitive_edges[j][1])
        test_data.append(test_q)
        
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data.to_csv(output_file, index=False)
    
    
def create_neg_edges_test(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x for x in data if len(x) != 0]
    
    edges = data[0].split('.')
    edges = [x.strip().split(' causes ') for x in edges if len(x) != 0]
    
    # store in a nicer format
    out_nodes = {}
    for edge in edges:
        if edge[0] in out_nodes:
            out_nodes[edge[0]].append(edge[1])
        else:
            out_nodes[edge[0]] = [edge[1]]
            
    # obtain all descendants            
    descendant_set = get_descendants(out_nodes)
    
    # create neg edges
    neg_edges = []
    for node in descendant_set.keys():
        for descendant in descendant_set[node]:
            neg_edges.append([descendant, node])
            
    test_data = []
    qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    
    for j in range(len(neg_edges)):
        test_q = qa_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
        test_data.append(test_q)
        
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data.to_csv(output_file, index=False)
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='convert synthetic to readable format for finetuning')
    parser.add_argument("--data_path", help="path to the synthetic txt dataset file")
    parser.add_argument("--save_path", help="path/name to save csv data")
    parser.add_argument("--convert_to_csv", action="store_true")
    parser.add_argument("--create_causal_graph_test", action="store_true")
    parser.add_argument("--create_transitivity_test", action="store_true")
    parser.add_argument("--create_neg_edges_test", action="store_true")
    
    args = parser.parse_args() 
    
    if args.convert_to_csv:
        convert_to_csv(args.data_path, args.save_path)
    
    if args.create_causal_graph_test:
        create_causal_graph_test(args.data_path, args.save_path)
    
    if args.create_transitivity_test:
        create_transitivity_test(args.data_path, args.save_path)
        
    if args.create_neg_edges_test:
        create_neg_edges_test(args.data_path, args.save_path)
    
    
    
    
