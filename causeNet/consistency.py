import json
import os
import openai
import time
import random
from nltk.tokenize import word_tokenize

random.seed(0)

openai.api_key = os.environ["OPENAI_API_KEY"]

def query_gpt3(query):

    response = openai.Completion.create(
        model="text-davinci-003",
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


def query_chatgpt(query):

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
      temperature=0,
      max_tokens=50
    )

    answer = response['choices'][0]['message']['content']
    return answer


def extract_answer(response):

    response = response.lower()
    response = word_tokenize(response)
    if "yes" in response:
        return 1
    elif "no" in response:
        return 0
    else:
        return -1


def graph_consistency(graph):

    consistent_subgraph = []
    consistent = True

    for edge in graph:

        cause = " ".join(edge[0].split('_'))
        effect = " ".join(edge[1].split('_'))

        query_correct_dir = "Can " + cause + " cause " + effect + "? Answer yes or no."
        query_incorrect_dir = "Can " + effect + " cause " + cause + "? Answer yes or no."

        response_correct_dir = query_chatgpt(query_correct_dir)
        response_incorrect_dir = query_chatgpt(query_incorrect_dir)

        prediction_correct_dir = extract_answer(response_correct_dir)
        prediction_incorrect_dir = extract_answer(response_incorrect_dir)

        if prediction_correct_dir == 1 and prediction_incorrect_dir == 0:
            consistent_subgraph.append(edge)
        else:
            consistent = False

    return consistent, consistent_subgraph


if __name__ == "__main__":

    # read the dataset
    data = []
    with open('causenet-precision.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Delete the source / support

    for j in range(len(data)):
        del data[j]['sources']
        del data[j]['support']

    # Store the data a nicer graph format
    out_nodes = {}
    in_nodes = {}
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


    # Filter out edges if there is also an edge in opposite direction
    # Also filter out cases where cause and effect are same!
    num_removed = 0
    for cause in out_nodes.keys():
        for effect in out_nodes[cause]:
            if cause == effect:
                out_nodes[cause].remove(effect)
                in_nodes[effect].remove(cause)

                num_removed += 1

            elif effect in out_nodes.keys() and cause in out_nodes[effect]:
                out_nodes[cause].remove(effect)
                out_nodes[effect].remove(cause)
                in_nodes[effect].remove(cause)
                in_nodes[cause].remove(effect)

                num_removed += 1


    print("Number of edge-pairs removed: ", num_removed)

    # Create a subgraph, centered around a manually selected concept
    # Restrict the number of children of any node to 2
    # Restrict the number of ancestor depth and children depth to 1 each

    num_tries = 0
    while True:
        child1 = random.choice(out_nodes['success'])
        child2 = random.choice(out_nodes['death'])
        parent1 = random.choice(in_nodes['success'])
        parent1_child = random.choice(out_nodes[parent1])
        subgraph = [(parent1, 'success'), ('success', child1)]
        consistent, consistent_subgraph = graph_consistency(subgraph)
        if consistent:
            break
        num_tries += 1
        if num_tries > 9:
            print("DONE: No consistent graph found in 10 tries")
            exit()

    
    print("DONE: Consistent graph found...")  
    print(consistent_subgraph)


