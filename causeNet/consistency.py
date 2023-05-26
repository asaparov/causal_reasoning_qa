import json
import os
import openai
import time
import random
from nltk.tokenize import word_tokenize
import argparse
import math
import pandas as pd
from tqdm import tqdm


openai.api_key = os.environ["OPENAI_API_KEY"]

def query_gpt3(query, model="text-davinci-003"):

    response = openai.Completion.create(
        model=model,
        prompt=query,
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"]
    )
    
    answer = response["choices"][0]["text"].strip().lower()
    return answer


def query_chatgpt(query, model="gpt-3.5-turbo"):

    response = openai.ChatCompletion.create(
      model=model,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
      temperature=0,
      max_tokens=50
    )

    answer = response['choices'][0]['message']['content']
    return answer


def extract_answer(response, mcq):

    response = response.lower()
    response = word_tokenize(response)
    if mcq and (response[0] == "a" or response[0] == "yes"):
        return 1
    elif mcq and (response[0] == "b" or response[0] == "no"):
        return 0
    elif "yes" in response:
        return 1
    elif "no" in response:
        return 0
    else:
        return -1


def graph_consistency(graph, model, mcq=False, k_shot=False, return_acc=False):

    consistent_subgraph = []
    consistent = True

    if "turbo" in args.model:
        query_fn = query_chatgpt
    else:
        query_fn = query_gpt3


    if return_acc:
        correct_dir_correct = 0
        incorrect_dir_correct = 0
        num_consistent = 0
        total_edges = 0


    if k_shot:
        assert args.mcq
        with open('8shot_mcq_prompt_minimal.txt', 'r') as f:
            few_shot_prompt = f.read()

    for e_num, edge in tqdm(enumerate(graph)):


        cause = " ".join(edge[0].split('_'))
        effect = " ".join(edge[1].split('_'))

        
        ## mcq
        if mcq:
            query_correct_dir = "Question: Can " + cause + " directly cause " + effect + "?\nA. Yes\nB. No\nAnswer:"
            query_incorrect_dir = "Question: Can " + effect + " directly cause " + cause + "?\nA. Yes\nB. No\nAnswer:"
            #query_correct_dir = "Question: Can " + cause + " really cause " + effect + "?\nA. Yes\nB. No\nAnswer:"
            #query_incorrect_dir = "Question: Can " + effect + " really cause " + cause + "?\nA. Yes\nB. No\nAnswer:"
        else:
            ## default format
            query_correct_dir = "Can " + cause + " cause " + effect + "? Answer yes or no."
            query_incorrect_dir = "Can " + effect + " cause " + cause + "? Answer yes or no."
            #query_correct_dir = "Is there a causal relationship between " + cause + " and " + effect + "? Answer yes or no."
            #query_incorrect_dir = "Is there a causal relationship between " + effect + " and " + cause + "? Answer yes or no."

        # If k_shot, append a prompt
        if k_shot:
            query_correct_dir = few_shot_prompt + query_correct_dir
            query_incorrect_dir = few_shot_prompt + query_incorrect_dir


        if e_num == 0:
            print(query_correct_dir)
            print('='*50)
            print(query_incorrect_dir)
            print('='*50)


        response_correct_dir = query_fn(query_correct_dir, model)
        response_incorrect_dir = query_fn(query_incorrect_dir, model)

        prediction_correct_dir = extract_answer(response_correct_dir, mcq)
        prediction_incorrect_dir = extract_answer(response_incorrect_dir, mcq)
        
        if prediction_correct_dir == 1 and prediction_incorrect_dir == 0:
            consistent_subgraph.append(edge)
        else:
            consistent = False

        if return_acc:
            if prediction_correct_dir == 1:
                correct_dir_correct += 1
            if prediction_incorrect_dir == 0:
                incorrect_dir_correct += 1
            if prediction_correct_dir == 1 and prediction_incorrect_dir == 0:
                num_consistent += 1
            total_edges += 1

    if return_acc:
        correct_dir_acc = correct_dir_correct/total_edges * 100
        incorrect_dir_acc = incorrect_dir_correct/total_edges * 100
        consistency = num_consistent/total_edges * 100
        return consistent, consistent_subgraph, correct_dir_acc, incorrect_dir_acc, consistency
    else:
        return consistent, consistent_subgraph


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--model', help='Non-chat Open AI model to use', default="text-davinci-003")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mcq', action='store_true')
    parser.add_argument('--k_shot', action='store_true')
    parser.add_argument('--random', action='store_true', help="select 100 random edges to evaluate consistency")
    parser.add_argument('--frequency', action='store_true', help="select edges based on freq and compute consistency")
    parser.add_argument('--subgraphs', action='store_true', help="find consistent subgraphs centered around some node")
    parser.add_argument('--tubingen', action='store_true', help='Check consistency on tubingen dataset')
    args = parser.parse_args()

    random.seed(args.seed)

    # read the dataset
    data = []
    with open('causenet-precision.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Delete the source / support

    for j in range(len(data)):
        data[j]['num_sources'] = len(data[j]['sources'])
        del data[j]['sources']
        del data[j]['support']

    # Store the data a nicer graph format
    out_nodes = {}
    in_nodes = {}
    freq = {}
    for j in range(len(data)):
        cause = data[j]['causal_relation']['cause']['concept']
        effect =  data[j]['causal_relation']['effect']['concept']
        if cause in out_nodes:
            out_nodes[cause].append(effect)
        else:
            out_nodes[cause] = [effect]

        if effect in in_nodes:
            in_nodes[effect].append(cause)
        else:
            in_nodes[effect] = [cause]

        # Use number of sources as proxy for freq
        freq[(cause, effect)] = data[j]['num_sources']


    # Filter out edges if there is also an edge in opposite direction
    # Also filter out cases where cause and effect are same!
    num_removed = 0
    for cause in out_nodes.keys():
        for effect in out_nodes[cause]:
            if cause == effect:
                out_nodes[cause].remove(effect)
                in_nodes[effect].remove(cause)
                del freq[(cause, effect)]

                num_removed += 1

            elif effect in out_nodes.keys() and cause in out_nodes[effect]:
                out_nodes[cause].remove(effect)
                out_nodes[effect].remove(cause)
                in_nodes[effect].remove(cause)
                in_nodes[cause].remove(effect)
                del freq[(cause, effect)]
                del freq[(effect, cause)]

                num_removed += 1


    print("Number of edge-pairs removed: ", num_removed)


    # Gather all edges
    all_edges = []
    for cause in out_nodes.keys():
        for effect in out_nodes[cause]:
            all_edges.append((cause, effect))

    print("Total edges in graph: ", len(all_edges))


    if args.tubingen:

        tubingen_data = pd.read_csv('tubingen.txt')
        all_tubingen_edges = []
        for j in range(len(tubingen_data)):
            if tubingen_data['direction'][j] == "right":
                all_tubingen_edges.append((tubingen_data['X'][j], tubingen_data['Y'][j]))
            elif tubingen_data['direction'][j] == "left":
                all_tubingen_edges.append((tubingen_data['Y'][j], tubingen_data['X'][j]))

        _, _, correct_dir_acc, incorrect_dir_acc, consistency = graph_consistency(all_tubingen_edges, model=args.model, mcq=args.mcq, k_shot=args.k_shot, return_acc=True)
        print("Correct direction accuracy: ", correct_dir_acc)
        print("Incorrect direction acccuracy: ", incorrect_dir_acc)
        print("Consistency score: ", consistency)



    if args.random:

        # Check consistency for 100 random edges
        random_subgraph = random.sample(all_edges, 100)
        #random_subgraph = all_edges
        _, _, correct_dir_acc, incorrect_dir_acc, consistency = graph_consistency(random_subgraph, model=args.model, mcq=args.mcq, k_shot=args.k_shot, return_acc=True)
        print("Correct direction accuracy: ", correct_dir_acc)
        print("Incorrect direction acccuracy: ", incorrect_dir_acc)
        print("Consistency score: ", consistency)


        # Write to out file
        if args.mcq and args.k_shot:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_100random_4shot_mcq.txt"
        elif args.mcq:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_100random_mcq.txt"
        elif args.k_shot:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_100random_4shot.txt"
        else:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_100random.txt"
        
        with open(out_file, "w") as f:
            f.write(str(correct_dir_acc) + "\n")
            f.write(str(incorrect_dir_acc))


    if args.frequency:

        # log scale
        log_freq_buckets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        freq_buckets = [math.exp(x) for x in log_freq_buckets]
        print(freq_buckets)
        
        # get edges in each bucket
        bucketed_edges = []
        for j in range(len(freq_buckets) - 1):
            current_edges = [e for e, v in freq.items() if v >= freq_buckets[j] and v < freq_buckets[j+1]]
            bucketed_edges.append(current_edges)

        # sample 40 random edges from each bucket
        random_edges = [random.sample(x, 40) for x in bucketed_edges]
        correct_dir_accs = []
        incorrect_dir_accs = []
        consistency_scores = []
        for j in range(len(freq_buckets) - 1):
            current_edges = random_edges[j]
            _, _, correct_dir_acc, incorrect_dir_acc, consistency = graph_consistency(current_edges, model=args.model, mcq=args.mcq, k_shot=args.k_shot, return_acc=True)
            correct_dir_accs.append(correct_dir_acc)
            incorrect_dir_accs.append(incorrect_dir_acc)
            consistency_scores.append(consistency)
            print("Bucket " + str(j) + " correct dir: ", correct_dir_acc)
            print("Bucket " + str(j) + " incorrect dir: ", incorrect_dir_acc)
            print("Bucket " + str(j) + " consistency: ", consistency)
            time.sleep(5)

        if args.mcq and args.k_shot:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_freq_4shot_mcq.txt"
        elif args.mcq:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_freq_mcq.txt"
        elif args.k_shot:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_freq_4shot.txt"
        else:
            out_file = 'out/' + args.model + "_seed" + str(args.seed) + "_freq.txt"
        
        with open(out_file, "w") as f:
            for j in range(len(correct_dir_accs)):
                f.write(str(correct_dir_accs[j]) + "\n")
                f.write(str(incorrect_dir_accs[j]) + "\n")
                f.write(str(consistency_scores[j]) + "\n")


    if args.subgraphs:

        # Create a subgraph, centered around a manually selected concept
        # Restrict the number of children of any node to 2
        # Restrict the number of ancestor depth and children depth to 1 each

        num_tries = 0
        while True:
            child1 = random.choice(out_nodes['stress'])
            child2 = random.choice(out_nodes['stress'])
            parent1 = random.choice(in_nodes['stress'])
            #parent1_child = random.choice(out_nodes[parent1])
            subgraph = [(parent1, 'stress') , ('stress', child1), ('stress', child2)]
            consistent, consistent_subgraph = graph_consistency(subgraph, args.model)
            if consistent:
                #print(consistent_subgraph)
                continue
            if len(consistent_subgraph) == 0:
                print(subgraph)
            num_tries += 1
            if num_tries > 10:
                print("DONE: No consistent graph found in 10 tries")
                exit()

    
        print("DONE: Consistent graph found...")  
        print(consistent_subgraph)


