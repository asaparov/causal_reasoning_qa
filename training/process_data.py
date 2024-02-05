import os
import sys
import csv
import argparse
import pandas as pd
import random
import re
from nltk import word_tokenize

def convert_to_csv(input_file, output_file, keep_causal_graph=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    # Remove the causal graph and negative edges from file
    if args.keep_causal_graph:
        scenarios = data[2:]
        graph = [x.strip() + "." for x in data[0].split('.')][:-1]
        data = graph + scenarios
    else:
        data = data[2:]
    
    data = pd.DataFrame(data, columns=['text'])
    data.to_csv(output_file, index=False)
    
    
def create_causal_graph_test(input_file, output_file, framing="default", options=False, ontology=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    causal_graph_edges = data[0].split('.')
    if ontology:
        causal_graph_edges = [x.strip().split(' causes ') for x in causal_graph_edges if len(x) != 0 and 'causes' in x and 'eventtype' in x]
    else:
        causal_graph_edges = [x.strip().split(' causes ') for x in causal_graph_edges if len(x) != 0 and 'causes' in x and 'eventtype' not in x]
    
    if options:
        test_data = {"option1": [], "option2": [], "answer": []}
        if framing == "default":
            option1_template = "{cause} can cause {effect}"
            option2_template = option1_template
        elif framing == "inverted":
            option1_template = "{cause} cannot cause {effect}"
            option2_template = option1_template
        elif framing == "counterfactual":
            option1_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen? {answer}"
            option2_template = option1_template
        elif framing == "indirect":
            option1_template = "{cause} can indirectly cause {effect}"
            option2_template = option1_template

        for j in range(len(causal_graph_edges)):
            if framing == "counterfactual":
                option1 = option1_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1], answer="yes")
                option2 = option1_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1], answer="no")
            elif framing == "default":
                option1 = option1_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1])
                option2 = option2_template.format(cause=causal_graph_edges[j][1], effect=causal_graph_edges[j][0])
                answer = "option1"
            elif framing == "inverted":
                option1 = option1_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1])
                option2 = option2_template.format(cause=causal_graph_edges[j][1], effect=causal_graph_edges[j][0])
                answer = "option2"
            elif framing == "indirect":
                option1 = option1_template.format(cause=causal_graph_edges[j][0], effect=causal_graph_edges[j][1])
                option2 = option2_template.format(cause=causal_graph_edges[j][1], effect=causal_graph_edges[j][0])
                answer = "option1"
            
            test_data["option1"].append(option1)
            test_data["option2"].append(option2)
            test_data["answer"].append(answer)

        test_data = pd.DataFrame(test_data)
        test_data.to_csv(output_file, index=False)
    else:
        test_data = []
        if framing == "default":
            qa_template = "Answer yes or no. Can {cause} cause {effect}?"
        elif framing == "inverted":
            qa_template = "Answer yes or no. {cause} cannot cause {effect} - is this correct?"
        elif framing == "counterfactual":
            qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"

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


def get_2_hop(graph):
    
    descendant_set = {}
    topological_order = iterative_topological_sort(graph)
    for node in topological_order:
        descendant_set[node] = []
        if node in graph:
            for child in graph[node]:
                if child in graph:
                    for grandchild in graph[child]:
                        descendant_set[node].append(grandchild)
                        
    return descendant_set
    

def create_transitivity_test(input_file, output_file, num_hops=-1, framing="default", options=False, ontology=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    edges = data[0].split('.')
    if ontology:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' in x]
    else:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' not in x]
        
    # store in a nicer format
    out_nodes = {}
    for edge in edges:
        if edge[0] in out_nodes:
            out_nodes[edge[0]].append(edge[1])
        else:
            out_nodes[edge[0]] = [edge[1]]
            
    # obtain all descendants 
    if num_hops == -1:
        descendant_set = get_descendants(out_nodes)
    elif num_hops == 2:
        descendant_set = get_2_hop(out_nodes)
            
    # Remove the original edges to create transitive set
    transitive_edges = []
    for node in descendant_set.keys():
        for descendant in descendant_set[node]:
            if [node, descendant] not in edges:
                transitive_edges.append([node, descendant])
                
    if options:        
        test_data = {"option1": [], "option2": [], "answer": []}
        if framing == "default":
            sent_template = "{cause} can cause {effect}"
        elif framing == "inverted":
            sent_template = "{cause} cannot cause {effect}"
        elif framing == "indirect":
            sent_template = "{cause} can indirectly cause {effect}"

        for j in range(len(transitive_edges)):
            option1 = sent_template.format(cause=transitive_edges[j][0], effect=transitive_edges[j][1])
            option2 = sent_template.format(cause=transitive_edges[j][1], effect=transitive_edges[j][0])
            if framing == "default":
                answer = "option1"
            elif framing == "inverted":
                answer = "option2"
            elif framing == "indirect":
                answer = "option1"
            test_data["option1"].append(option1)
            test_data["option2"].append(option2)
            test_data["answer"].append(answer)

        test_data = pd.DataFrame(test_data)
        test_data.to_csv(output_file, index=False)  
    else:
                    
        test_data = []
        if framing == "default":
            qa_template = "Answer yes or no. Can {cause} cause {effect}?"
        elif framing == "inverted":
            qa_template = "Answer yes or no. {cause} cannot cause {effect} - is this correct?"
        elif framing == "counterfactual":
            qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"

        for j in range(len(transitive_edges)):
            test_q = qa_template.format(cause=transitive_edges[j][0], effect=transitive_edges[j][1])
            test_data.append(test_q)

        test_data = pd.DataFrame(test_data, columns=['text'])
        test_data.to_csv(output_file, index=False)
    
    
def create_neg_edges_test(input_file, output_file, framing="default", ontology=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    edges = data[0].split('.')
    if ontology:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' in x]
    else:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' not in x]
    
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
    if framing == "default":
        qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    elif framing == "inverted":
        qa_template = "Answer yes or no. {cause} cannot cause {effect} - is this correct?"
    elif framing == "counterfactual":
        qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"
    
    for j in range(len(neg_edges)):
        test_q = qa_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
        test_data.append(test_q)
        
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data.to_csv(output_file, index=False)
    
    
def create_norelation_test(input_file, output_file, framing="default", ontology=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    edges = data[0].split('.')
    if ontology:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' in x]
    else:
        edges = [x.strip().split(' causes ') for x in edges if len(x) != 0 and 'causes' in x and 'eventtype' not in x]
    
    
    # store in a nicer format
    out_nodes = {}
    for edge in edges:
        if edge[0] in out_nodes:
            out_nodes[edge[0]].append(edge[1])
        else:
            out_nodes[edge[0]] = [edge[1]]
            
    # obtain all descendants            
    descendant_set = get_descendants(out_nodes)
    
    # chose pairs of nodes where none is descendant of the other
    norelation = []
    for node1 in descendant_set.keys():
        for node2 in descendant_set.keys():
            if node1 not in descendant_set[node2] and node2 not in descendant_set[node1] and node1 != node2:
                norelation.append([node1, node2])
      
    
    test_data = []
    if framing == "default":
        qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    elif framing == "counterfactual":
        qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"
    
    for j in range(len(norelation)):
        test_q = qa_template.format(cause=norelation[j][0], effect=norelation[j][1])
        test_data.append(test_q)
        
    test_data = pd.DataFrame(test_data, columns=['text'])
    test_data.to_csv(output_file, index=False)
    
    
def create_balanced_data(input_file, output_file):
    
    # Balance the scenarios post-hoc so that 'yes' and 'no' are balanced
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    # Remove the causal graph and negative edges from file
    data = data[2:]
    
    # Find indices which only contain 'yes' or 'no' or contain both
    yes_idx = []
    no_idx = []
    mixed_idx = []
    mixed_diff = []
    total_yes = 0
    total_no = 0
    for j in range(len(data)):
        yes_count = data[j].count('yes.')
        no_count = data[j].count('no.')
        total_yes += yes_count
        total_no += no_count
        if yes_count > 0 and no_count == 0:
            yes_idx.append(j)
        elif yes_count == 0 and no_count > 0:
            no_idx.append(j)
        elif yes_count > 0 and no_count > 0:
            mixed_idx.append(j)
            mixed_diff.append(yes_count - no_count)
            
    assert total_yes > total_no    
    
    # Remove some example which only have 'yes'
    random.shuffle(yes_idx)
    remove_idx = []
    for idx in yes_idx:
        remove_idx.append(idx)
        total_yes -= data[idx].count('yes.')
        if total_yes <= total_no:
            break
            
    # If that's still not enough, remove some 'mixed' examples
    if total_yes > total_no:
        # Sort the idx in descending order of difference of 'yes' and 'no' so as to remove the min. no of examples 
        mixed_idx = [idx for _, idx in sorted(zip(mixed_diff, mixed_idx), reverse=True)]
        for idx in mixed_idx:
            remove_idx.append(idx)
            total_yes -= data[idx].count('yes.')
            total_no -= data[idx].count('no.')
            if total_yes <= total_no:
                break
                
    data_balanced = [item for idx, item in enumerate(data) if idx not in remove_idx]
    data_balanced = pd.DataFrame(data_balanced, columns=['text'])
    data_balanced.to_csv(output_file, index=False)


def create_type_generalization_test(input_file, output_file, source=True, target=False, ood_events=False, framing="default", options=False):
    
    with open(input_file, 'r') as f:
        data = f.readlines()
        
    data = [x.strip() for x in data]
    data = [x.lower() for x in data if len(x) != 0]
    
    graph_info = data[0].split('.')
    graph_edges = [x.strip().split(' causes ') for x in graph_info if len(x) != 0 and 'causes' in x and 'eventtype' not in x]
    type_edges = [x.strip().split(' causes ') for x in graph_info if len(x) != 0 and 'causes' in x and 'eventtype' in x]
    
    # store in a nicer format
    out_nodes = {}
    for edge in graph_edges:
        if edge[0] in out_nodes:
            out_nodes[edge[0]].append(edge[1])
        else:
            out_nodes[edge[0]] = [edge[1]]
            
    descendant_set = get_descendants(out_nodes)
    
    # create a type to event dict
    event_to_type = {}
    type_to_event = {}
    for x in graph_info:
        if 'type of' in x:
            x = x.strip().split(' ')
            if x[-1] in type_to_event:
                type_to_event[x[-1]].append(x[0])
            else:
                type_to_event[x[-1]] = [x[0]]
                
    # remove the events which are used in type edges (we want non-causal types only in this test)
    for edge in type_edges:
        _ = type_to_event.pop(edge[0], None)
        _ = type_to_event.pop(edge[1], None)
    
    # create event to type afterwards so as to not include causal types
    for node_type in type_to_event.keys():
        for node in type_to_event[node_type]:
            if node in event_to_type:
                event_to_type[node].append(node_type)
            else:
                event_to_type[node] = [node_type]
                
    # ood events -- create new events as neighbours instead of existing events
    if ood_events:

        # Gather all events
        all_nodes = set()
        all_nodes.update(list(out_nodes.keys()))
        all_nodes.update([x for val in out_nodes.values() for x in val])
        all_nodes = [int(x[5:]) for x in list(all_nodes)]
        max_event = max(all_nodes)
        
        neg_edges = []
        ref_edges = []
        prefixes = []
        for edge in graph_edges:
            if source:
                for edge_type in event_to_type[edge[0]]:
                    new_event = 'event' + str(max_event + 1)
                    neg_edges.append([new_event, edge[1]])
                    ref_edges.append([edge[0], edge[1]])
                    prefixes.append(new_event + " is a type of " + edge_type + ". ")
                    max_event += 1
                    
            if target:
                for edge_type in event_to_type[edge[1]]:
                    new_event = 'event' + str(max_event + 1)
                    neg_edges.append([edge[0], new_event])
                    ref_edges.append([edge[0], edge[1]])
                    prefixes.append(new_event + " is a type of " + edge_type + ". ")
                    max_event += 1
        
    
    else:
    
        # create neg edges -- i.e. edges between nodes of same non-causal type as an an actual edge
        # make sure the new edge is not possible (e.g. actual edge or transitivity)
        neg_edges = []
        ref_edges = []
        prefixes = []
        for edge in graph_edges:
            if source:
                for edge_type in event_to_type[edge[0]]:
                    for neighbour_node in type_to_event[edge_type]:
                        if neighbour_node != edge[0] and (neighbour_node not in descendant_set or edge[1] not in descendant_set[neighbour_node]):
                            neg_edges.append([neighbour_node, edge[1]])
                            ref_edges.append([edge[0], edge[1]])
                            prefixes.append("")

            if target:
                for edge_type in event_to_type[edge[1]]:
                    for neighbour_node in type_to_event[edge_type]:
                        if neighbour_node != edge[1] and neighbour_node not in descendant_set[edge[0]]:
                            neg_edges.append([edge[0], neighbour_node])
                            ref_edges.append([edge[0], edge[1]])
                            prefixes.append("")

    
    if options:
        test_data = {"option1": [], "option2": [], "answer": []}
        if framing == "default":
            option1_template = "{cause} can cause {effect}"
            option2_template = "{cause} cannot cause {effect}"
        elif framing == "reference":
            option1_template = "{cause} can cause {effect}"
            option2_template = "{cause} can cause {effect}"

        for j in range(len(neg_edges)):
            if framing == "default":
                option1 = prefixes[j] + option1_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
                option2 = prefixes[j] + option2_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
                answer = "option2"
            elif framing == "reference":
                option1 = prefixes[j] + option1_template.format(cause=ref_edges[j][0], effect=ref_edges[j][1])
                option2 = prefixes[j] + option2_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
                answer = "option1"
                
            test_data["option1"].append(option1)
            test_data["option2"].append(option2)
            test_data["answer"].append(answer)
        
        test_data = pd.DataFrame(test_data)
        test_data.to_csv(output_file, index=False)  
    else:
        
        test_data = []
        if framing == "default":
            qa_template = "Answer yes or no. Can {cause} cause {effect}?"
        elif framing == "inverted":
            qa_template = "Answer yes or no. {cause} cannot cause {effect} - is this correct?"
        elif framing == "counterfactual":
            qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"

        for j in range(len(neg_edges)):
            test_q = qa_template.format(cause=neg_edges[j][0], effect=neg_edges[j][1])
            test_data.append(test_q)

        test_data = pd.DataFrame(test_data, columns=['text'])
        test_data.to_csv(output_file, index=False)
    
    
def convert_to_counterfactuals(input_file, output_file, framing="default", options=False):
    
    data = pd.read_csv(input_file)
    c_data = []
    if framing == "default":
        qa_template = "Answer yes or no. {cause} happened. {effect} happened. If {cause} did not happen, and {effect} has no other causes, would {effect} happen?"
    elif framing == "inverted":
        qa_template = "Answer yes or no. {effect} happened. {cause} happened. If {effect} did not happen, and {cause} has no other causes, would {cause} happen?"
    for j in range(len(data)):
        ex = data['option1'][j]
        ex_events = [x for x in ex.split(" ") if 'event' in x]
        c_ex = qa_template.format(cause=ex_events[0], effect=ex_events[1])
        c_data.append(c_ex)
        
    c_data = pd.DataFrame(c_data, columns=["text"])
    c_data.to_csv(output_file, index=False)
    
def convert_to_qa(input_file, output_file, framing="default"):
    
    data = pd.read_csv(input_file)
    new_data = []
    
    if framing == "default":
        qa_template = "Answer yes or no. Can {cause} cause {effect}?"
    elif framing == "inverted":
        qa_template = "Answer yes or no. {cause} cannot cause {effect} - is this correct?"
        
    for j in range(len(data)):
        ex = data['option1'][j]
        ex_events = [x for x in ex.split(" ") if 'event' in x]
        new_ex = qa_template.format(cause=ex_events[0], effect=ex_events[1])
        new_data.append(new_ex)
        
    new_data = pd.DataFrame(new_data, columns=['text'])
    new_data.to_csv(output_file, index=False)
    

def convert_to_options(input_file, output_file, framing="default"):
    
    data = pd.read_csv(input_file)
    new_data = {"option1": [], "option2": [], "answer": []}
    
    if framing == "default":
        option1_template = "{cause} can cause {effect}"
        option2_template = "{cause} cannot cause {effect}"
    elif framing == "inverted":
        option1_template = "{cause} cannot cause {effect}"
        option2_template = "{cause} can cause {effect}"
        
    for j in range(len(data)):
        ex = data['text'][j]
        ex = word_tokenize(ex)
        ex_events = [x for x in ex if 'event' in x]
        if framing == "default":
            new_ex_option1 = option1_template.format(cause=ex_events[0], effect=ex_events[1])
            new_ex_option2 = option2_template.format(cause=ex_events[0], effect=ex_events[1])
            answer = "option1"
        new_data["option1"].append(new_ex_option1)
        new_data["option2"].append(new_ex_option2)
        new_data["answer"].append(answer)
        
    new_data = pd.DataFrame(new_data)
    new_data.to_csv(output_file, index=False)
        
        
    
def change_framing(input_file, output_file, framing):
    
    data = pd.read_csv(input_file)
    
    # Assumption: input data has this format
    default_template = "{cause} can cause {effect}"
    
    if framing == "framing0":
        sent_template = "{cause} can cause {effect}"
    elif framing == "framing1":
        sent_template = "{effect} can be caused by {cause}"
    elif framing == "framing2":
        sent_template = "{effect} is caused by {cause}"
    elif framing == "framing3":
        sent_template = "{cause} causally affects {effect}"
    elif framing == "framing4":
        sent_template = "{cause} can lead to {effect}"
    elif framing == "framing5":
        sent_template = "{cause} is related to {effect}"
    elif framing == "framing6":
        sent_template = "{cause} {effect}"
    elif framing == "framing7":
        sent_template = "{effect} is causally affected by {cause}"
    elif framing == "framing8":
        sent_template = "{cause} is complementary to {effect}"
    elif framing == "framing9":
        sent_template = "{cause} happened next to {effect}"
    # Next three are opposite of 5, 8, 9
    elif framing == "framing10":
        sent_template = "{effect} is related to {cause}"
    elif framing == "framing11":
        sent_template = "{effect} is complementary to {cause}"
    elif framing == "framing12":
        sent_template = "{effect} happened next to {cause}"
        

        
        
    test_data = {"option1": [], "option2": [], "answer": []}
    for j in range(len(data)):
        m = re.match(r'(.*) can cause (.*)', data['option1'][j])
        cause, effect = m.group(1), m.group(2)
        option1 = sent_template.format(cause=cause, effect=effect)
        option2 = sent_template.format(cause=effect, effect=cause)
        answer = "option1"
        test_data["option1"].append(option1)
        test_data["option2"].append(option2)
        test_data["answer"].append(answer)

    test_data = pd.DataFrame(test_data)
    test_data.to_csv(output_file, index=False)

    
def change_framing_binary(input_file, output_file, framing):
    
    data = pd.read_csv(input_file)
    
    # Assumption: input data has this format (2nd option in 'binary' is {cause} cannot cause {effect}
    default_template = "{cause} can cause {effect}"
    
    if framing == "framing0":
        option1_template = "{cause} can cause {effect}"
        option2_template = "{cause} cannot cause {effect}"
    elif framing == "framing1":
        option1_template = "{effect} can be caused by {cause}"
        option2_template = "{effect} cannot be caused by {cause}"
    elif framing == "framing2":
        option1_template = "{effect} is caused by {cause}"
        option2_template = "{effect} is not caused by {cause}"
    elif framing == "framing3":
        option1_template = "{cause} causally affects {effect}"
        option2_template = "{cause} does not causally affect {effect}"
    elif framing == "framing4":
        option1_template = "{cause} can lead to {effect}"
        option2_template = "{cause} cannot lead to {effect}"
    elif framing == "framing5":
        option1_template = "{cause} is related to {effect}"
        option2_template = "{cause} is not related to {effect}"
    elif framing == "framing6":
        raise NotImplementedError
    elif framing == "framing7":
        option1_template = "{effect} is causally affected by {cause}"
        option2_template = "{effect} is not causally affected by {cause}"

        
        
    test_data = {"option1": [], "option2": [], "answer": []}
    for j in range(len(data)):
        m = re.match(r'(.*) can cause (.*)', data['option1'][j])
        cause, effect = m.group(1), m.group(2)
        option1 = option1_template.format(cause=cause, effect=effect)
        option2 = option2_template.format(cause=cause, effect=effect)
        answer = data["answer"][j]
        test_data["option1"].append(option1)
        test_data["option2"].append(option2)
        test_data["answer"].append(answer)

    test_data = pd.DataFrame(test_data)
    test_data.to_csv(output_file, index=False)


    
def change_options(input_file, output_file, framing):
    
    data = pd.read_csv(input_file)
    
    if framing == "binary":
        option1_template = "{cause} can cause {effect}"
        option2_template = "{cause} cannot cause {effect}"
    #elif framing == "control":
    #    option1_template = "{cause} can cause {effect}"
     
    test_data = {"option1": [], "option2": [], "answer": []}
    for j in range(len(data)):
        m = re.match(r'(.*) can cause (.*)', data['option1'][j])
        cause, effect = m.group(1), m.group(2)
        option1 = option1_template.format(cause=cause, effect=effect)
        option2 = option2_template.format(cause=cause, effect=effect)
        answer = "option1"
        test_data["option1"].append(option1)
        test_data["option2"].append(option2)
        test_data["answer"].append(answer)

    test_data = pd.DataFrame(test_data)
    test_data.to_csv(output_file, index=False)
    

def convert_to_three_way(input_file, output_file, framing):
    """
    Convert from the two option format to three options.
    Each option can have multiple implications separated by ` ; '
    """
    
    data = pd.read_csv(input_file)
    
    if framing == "framing1":
        sent_template = "{effect} can be caused by {cause}"
    elif framing == "framing2":
        sent_template = "{effect} is caused by {cause}"
    elif framing == "framing3":
        sent_template = "{cause} causally affects {effect}"
    elif framing == "framing4":
        sent_template = "{cause} can lead to {effect}"
    
    # 'can cause' might not be appropriate framing for three way
    if framing == "default":
        option1_template = "{cause} causes {effect} ; {effect} does not cause {cause}"
        option2_template = "{effect} causes {cause} ; {cause} does not cause {effect}"
        option3_template = "{cause} does not cause {effect} ; {effect} does not cause {cause}"
    elif framing == "framing0":
        option1_template = "{cause} can cause {effect} ; {effect} cannot cause {cause}"
        option2_template = "{effect} can cause {cause} ; {cause} cannot cause {effect}"
        option3_template = "{cause} cannot cause {effect} ; {effect} cannot cause {cause}"
    elif framing == "framing1":
        option1_template = "{effect} can be caused by {cause} ; {cause} cannot be caused by {effect}"
        option2_template = "{cause} can be caused by {effect} ; {effect} cannot be caused by {cause}"
        option3_template = "{cause} cannot be caused by {effect} ; {effect} cannot be caused by {cause}"
    elif framing == "framing2":
        option1_template = "{effect} is caused by {cause} ; {cause} is not caused by {effect}"
        option2_template = "{cause} is caused by {effect} ; {effect} is not caused by {cause}"
        option3_template = "{cause} is not caused by {effect} ; {effect} is not caused by {cause}"
    elif framing == "framing3":
        option1_template = "{cause} causally affects {effect} ; {effect} does not causally affect {cause}"
        option2_template = "{effect} causally affects {cause} ; {cause} does not causally affect {effect}"
        option3_template = "{effect} does not causally affect {cause} ; {cause} does not causally affect {effect}"
    elif framing == "framing4":
        option1_template = "{cause} can lead to {effect} ; {effect} cannot lead to {cause}"
        option2_template = "{effect} can lead to {cause} ; {cause} cannot lead to {effect}"
        option3_template = "{effect} cannot lead to {cause} ; {cause} cannot lead to {effect}"
    elif framing == "framing5":
        raise NotImplementedError
    elif framing == "framing6":
        raise NotImplementedError
    elif framing == "framing7":
        option1_template = "{effect} is causally affected by {cause} ; {cause} is not causally affected by {effect}"
        option2_template = "{cause} is causally affected by {effect} ; {effect} is not causally affected by {cause}"
        option3_template = "{cause} is not causally affected by {effect} ; {effect} is not causally affected by {cause}"
        
        
    test_data = {"option1": [], "option2": [], "option3": [], "answer": []}
    for j in range(len(data)):
        assert data["answer"][j] == "option2"
        m = re.match(r'(.*) can cause (.*)', data['option1'][j])
        cause, effect = m.group(1), m.group(2)
        option1 = option1_template.format(cause=cause, effect=effect)
        option2 = option2_template.format(cause=cause, effect=effect)
        option3 = option3_template.format(cause=cause, effect=effect)
        answer = "option3"
        #answer = data["answer"][j]
        test_data["option1"].append(option1)
        test_data["option2"].append(option2)
        test_data["option3"].append(option3)
        test_data["answer"].append(answer)
        
    test_data = pd.DataFrame(test_data)
    test_data.to_csv(output_file, index=False)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='convert synthetic to readable format for finetuning')
    parser.add_argument("--data_path", help="path to the synthetic txt dataset file")
    parser.add_argument("--save_path", help="path/name to save csv data")
    parser.add_argument("--convert_to_csv", action="store_true")
    parser.add_argument("--create_causal_graph_test", action="store_true")
    parser.add_argument("--create_transitivity_test", action="store_true")
    parser.add_argument("--create_neg_edges_test", action="store_true")
    parser.add_argument("--create_norelation_test", action="store_true")
    parser.add_argument("--create_balanced_data", action="store_true")
    parser.add_argument("--create_type_generalization_test", action="store_true")
    parser.add_argument("--framing", type=str, default="default")
    parser.add_argument("--options", action="store_true")
    parser.add_argument("--keep_causal_graph", action="store_true")
    parser.add_argument("--ontology", action="store_true", help="create graph/edges over types instead of events")
    parser.add_argument("--source", action="store_true", help="source type generalization")
    parser.add_argument("--target", action="store_true", help="target type generalization")
    parser.add_argument("--ood_events", action="store_true", help="use new events for src/tgt type generalization tests")
    parser.add_argument("--hops", type=int, default=-1)
    parser.add_argument("--convert_to_counterfactuals", action="store_true", help="convert options to counterfactual questions")
    parser.add_argument("--change_framing", action="store_true", help="change from the default framing to some target framing of the options")
    parser.add_argument("--change_framing_binary", action="store_true", help="change from the default framing to some target framing of the options for binary options")
    parser.add_argument("--convert_to_qa", action="store_true", help="convert from options to simple yes/no questions")
    parser.add_argument("--convert_to_options", action="store_true", help="convert from qa to options")
    parser.add_argument("--change_options", action="store_true", help="change the reference options to something else")
    parser.add_argument("--convert_to_three_way", action="store_true", help="convert to three options where each has two implications")
    
    args = parser.parse_args()
    
    #seed
    random.seed(0)
    
    if args.convert_to_csv:
        convert_to_csv(args.data_path, args.save_path, args.keep_causal_graph)
    
    if args.create_causal_graph_test:
        create_causal_graph_test(args.data_path, args.save_path, args.framing, args.options, args.ontology)
    
    if args.create_transitivity_test:
        create_transitivity_test(args.data_path, args.save_path, args.hops, args.framing, args.options, args.ontology)
        
    if args.create_neg_edges_test:
        create_neg_edges_test(args.data_path, args.save_path, args.framing)
        
    if args.create_norelation_test:
        create_norelation_test(args.data_path, args.save_path, args.framing)
               
    if args.create_balanced_data:
        create_balanced_data(args.data_path, args.save_path)
        
    if args.create_type_generalization_test:
        create_type_generalization_test(args.data_path, args.save_path, args.source, args.target, args.ood_events, args.framing, args.options)
    
    if args.convert_to_counterfactuals:
        convert_to_counterfactuals(args.data_path, args.save_path, args.framing, args.options)
        
    if args.change_framing:
        change_framing(args.data_path, args.save_path, args.framing)
        
    if args.change_framing_binary:
        change_framing_binary(args.data_path, args.save_path, args.framing)
    
    if args.convert_to_qa:
        convert_to_qa(args.data_path, args.save_path, args.framing)
        
    if args.convert_to_options:
        convert_to_options(args.data_path, args.save_path, args.framing)
        
    if args.change_options:
        change_options(args.data_path, args.save_path, args.framing)
    
    if args.convert_to_three_way:
        convert_to_three_way(args.data_path, args.save_path, args.framing)
    
    
    
