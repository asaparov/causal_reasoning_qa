import gpt3
import json
import getpass

model_name = "text-davinci-003"

with open("counterfactuals.json", "r") as f:
	data = json.load(f)

option_letters = ['A', 'B', 'C', 'D', 'E']
gpt_api_key = getpass.getpass(prompt='Enter OpenAI API Key:')

def query_example(question, options):
	prompt = example['question']
	for i in range(len(example['options'])):
		prompt += '\n' + option_letters[i] + '. ' + example['options'][i]
	prompt += '\n\nAnswer:'
	print(prompt)
	prediction, logprobs = gpt3.predict(gpt_api_key, model_name, prompt)
	prediction = prediction.strip()
	print(prediction + '\n\n')
	if prediction.lower().startswith('The correct answer is '):
		prediction = prediction[len('The correct answer is '):]
	elif prediction.lower().startswith('The answer is '):
		prediction = prediction[len('The answer is '):]
	print("Predicted answer: " + prediction[0].upper())
	return prediction[0].upper(), prediction

def reverse_answer(answer, num_options):
	# 'A' -> 'D' if num_options == 4
	# 'A' -> 'C' if num_options == 3
	# 'B' -> 'B' if num_options == 3
	i = option_letters.index(answer)
	return option_letters[num_options - i - 1]

model_outputs = []
for example in data:
	predicted_answer, model_output = query_example(example['question'], example['options'])
	example['model_output'] = model_output
	example['predicted_answer'] = predicted_answer
	example['correct'] = predicted_answer == example['answer']
	model_outputs.append(example)

	example = dict(example)
	options_copy = list(example['options'])
	options_copy.reverse()
	example['options'] = options_copy
	example['answer'] = reverse_answer(example['answer'], len(example['options']))
	predicted_answer, model_output = query_example(example['question'], example['options'])
	example['model_output'] = model_output
	example['predicted_answer'] = predicted_answer
	example['correct'] = predicted_answer == example['answer']
	model_outputs.append(example)

with open('counterfactual_outputs/' + model_name + '.json', 'w') as f:
	json.dump(model_outputs, f, indent=1)

consistent_subset = [ex for ex in model_outputs if ex['consistent']]
inconsistent_subset = [ex for ex in model_outputs if not ex['consistent']]
forward_subset = [ex for ex in model_outputs if ex['direction'] == 1]
backward_subset = [ex for ex in model_outputs if ex['direction'] == -1]
forward_consistent_subset = [ex for ex in model_outputs if ex['direction'] == 1 and ex['consistent']]
backward_consistent_subset = [ex for ex in model_outputs if ex['direction'] == -1 and ex['consistent']]

print('Total accuracy: {}/{}'.format(len([ex for ex in model_outputs if ex['correct']]), len(model_outputs)))
print('Accuracy on consistent edges: {}/{}'.format(len([ex for ex in consistent_subset if ex['correct']]), len(consistent_subset)))
print('Accuracy on inconsistent edges: {}/{}'.format(len([ex for ex in inconsistent_subset if ex['correct']]), len(inconsistent_subset)))
print('Accuracy on forward edges: {}/{}'.format(len([ex for ex in forward_subset if ex['correct']]), len(forward_subset)))
print('Accuracy on reverse edges: {}/{}'.format(len([ex for ex in backward_subset if ex['correct']]), len(backward_subset)))
print('Accuracy on forward consistent edges: {}/{}'.format(len([ex for ex in forward_consistent_subset if ex['correct']]), len(forward_consistent_subset)))
print('Accuracy on reverse consistent edges: {}/{}'.format(len([ex for ex in backward_consistent_subset if ex['correct']]), len(backward_consistent_subset)))
