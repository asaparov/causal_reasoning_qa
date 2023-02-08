from random import randrange, choice, sample, seed
import itertools
import gpt3
import getpass
import json

print_debug_info = False

PRESENT_TENSE = { "turn":"turns", "be":"is" }
PAST_TENSE = { "turn":"turned", "be":"was" }
AUXILIARY_VERBS = { "be" }

class Node(object):
	def __init__(self):
		self.actor = None
		self.color = None
		self.children = []
		self.parents = []

	def to_clause(self, verb, past=False, negate=False, finite=True, invert=False, adverb=None):
		if adverb == None:
			adverb = ""
		else:
			adverb = " " + adverb
		if verb.startswith("must "):
			inflected_verb = verb
		elif not finite:
			inflected_verb = "to " + verb
		elif past:
			inflected_verb = PAST_TENSE[verb]
		else:
			inflected_verb = PRESENT_TENSE[verb]
		color_np = ("not " + self.color if negate else self.color)
		if invert:
			return inflected_verb + " the " + self.actor + adverb + " " + color_np
		else:
			return "the " + self.actor + " " + inflected_verb + adverb + " " + color_np

	def __str__(self):
		return self.actor

	def __repr__(self):
		return self.actor

def generate_graph(num_vertices, max_num_parents, max_cyclic_edges):
	vertices = []
	for i in range(num_vertices):
		vertices.append(Node())

	# sample a random DAG
	num_sources = choice([1, 2])
	for i in range(num_sources, num_vertices):
		# sample the number of parent vertices
		if choice([True, False]):
			num_parents = 1
		else:
			num_parents = randrange(1, max_num_parents)
		num_parents = min(num_parents, i)

		for parent_id in sample(range(i), num_parents):
			vertices[parent_id].children.append(vertices[i])
			vertices[i].parents.append(vertices[parent_id])

	# add random cycles
	if max_cyclic_edges != 0:
		num_cyclic_edges = randrange(max_cyclic_edges)
		for i in range(num_cyclic_edges):
			edges = sample(range(num_vertices), 2)
			if edges[0] < edges[1]:
				temp = edges[0]
				edges[0] = edges[1]
				edges[1] = temp
			vertices[edges[0]].children.append(vertices[edges[1]])
			vertices[edges[1]].parents.append(vertices[edges[0]])

	return vertices

def capitalize(sentence):
	return sentence[0].upper() + sentence[1:]

def merge_paths(paths):
	merged = paths[0]
	for path in paths[1:]:
		for edge in path:
			if edge in merged:
				continue
			merged.append(edge)
	return merged

# TODO: this is for debugging; remove calls to this function when not debugging
def are_unique(paths):
	for i in range(len(paths)):
		for j in range(len(paths[i])):
			for k in range(j):
				if paths[i][j] == paths[i][k]:
					import pdb; pdb.set_trace()
					print('found duplicate steps in paths[{}] at indices {} and {}'.format(i, j, k))
					return False
		for j in range(i):
			if paths[i] == paths[j]:
				import pdb; pdb.set_trace()
				print('found duplicate proofs at indices {} and {}'.format(i, j))
				return False
	return True

def enumerate_hyperpaths(hypergraph, source, target, hypotheses, stack=[]):
	if source == target:
		return [[]]

	# enumerate all hyperpaths (NOTE: this is exponential, so it could be very costly for large/dense graphs)
	paths = []
	hyperedges_to_visit = []
	for (hypothesis_set, hyperedges) in hypergraph[target][1]:
		if hypothesis_set.issubset(hypotheses):
			hyperedges_to_visit.extend([e for e in hyperedges if e not in hyperedges_to_visit])
	for hyperedge in hyperedges_to_visit:
		if type(hyperedge) == list:
			if any([src in stack for src in hyperedge]):
				continue
			subpath_arrays = [enumerate_hyperpaths(hypergraph, source, src, hypotheses, stack=stack+[target]) for src in hyperedge]
			for path_combination in itertools.product(*subpath_arrays):
				merged = merge_paths(path_combination)
				if hyperedge not in merged:
					paths.append(merged + [hyperedge])
		elif type(hyperedge) == tuple:
			if hyperedge[-1] in stack:
				continue
			# this is a proof-by-cases step
			proofs_of_disjunction = enumerate_hyperpaths(hypergraph, source, hyperedge[-1], hypotheses, stack=stack+[target])
			proofs_of_disjunction = [proof for proof in proofs_of_disjunction if hyperedge[-1] not in proof and tuple(hyperedge[-1].parents) not in proof]
			for proof in proofs_of_disjunction:
				proof.append(hyperedge[-1])
				proof.append(tuple(hyperedge[-1].parents))
			proofs_of_cases = []
			for i in range(len(hyperedge) - 1):
				proof_list = []
				for case_hyperedge in hyperedge[i]:
					if type(case_hyperedge) == Node and case_hyperedge not in stack:
						case_hypotheses = set(hypotheses)
						case_hypotheses.add(hyperedge[-1].parents[i])
						case_proofs = enumerate_hyperpaths(hypergraph, hyperedge[-1].parents[i], case_hyperedge, case_hypotheses, stack=stack+[target])
						proof_list.extend([proof for proof in case_proofs if proof not in proof_list])
					are_unique(proof_list)
				if len(hyperedge[i]) == 0:
					proof_list = [[]]
				if target == hyperedge[-1].parents[i]:
					if [] not in proof_list:
						proof_list.append([])
				proofs_of_cases.append(proof_list)
			for combination in itertools.product(proofs_of_disjunction, *proofs_of_cases):
				proof = []
				subproofs_overlap = False
				for subproof in combination:
					if any([step in proof for step in subproof]):
						subproofs_overlap = True
						break
					proof.extend(subproof)
				if not subproofs_overlap:
					are_unique([proof])
				if not subproofs_overlap and proof not in paths:
					paths.append(proof)
		elif hyperedge in hypergraph and hyperedge not in stack:
			subpaths = enumerate_hyperpaths(hypergraph, source, hyperedge, hypotheses, stack=stack+[target])
			paths.extend([path + [hyperedge] for path in subpaths if hyperedge not in path and (path + [hyperedge]) not in paths])
		elif hyperedge == source:
			paths.append([hyperedge])
		are_unique(paths)
	return paths

def add_hypothesis_set(hypotheses, new_set, proof_edge, is_changed, debug_msg):
	# check if the new hypothesis set already exists in `hypotheses`
	for (existing_set, proof_edges) in hypotheses:
		if existing_set == new_set:
			if proof_edge != None and proof_edge not in proof_edges:
				if print_debug_info and debug_msg != None:
					print(debug_msg)
				proof_edges.append(proof_edge)
			return hypotheses, is_changed

	# add the new hypothesis set to `hypotheses`
	if print_debug_info and debug_msg != None:
		print(debug_msg)
	hypotheses.append((new_set, [proof_edge] if proof_edge != None else []))
	return hypotheses, True

def reduce_hypothesis_set(hypotheses):
	new_hypotheses = []
	for hypothesis_set in hypotheses:
		new_hypotheses, _ = add_hypothesis_set(new_hypotheses, hypothesis_set, None, False, None)
	return new_hypotheses

def add_proof_edge(proof_edges, hypotheses, new_edge):
	if hypotheses not in proof_edges:
		proof_edges[hypotheses] = new_edge
	elif new_edge not in proof_edges[hypotheses]:
			proof_edges[hypotheses].append(new_edge)

def solve_causal_query(start_vertex, start_value, target_vertex):
	disjunctions = []
	visited = []
	queue = [start_vertex]
	while len(queue) != 0:
		next_vertex = queue.pop()
		visited.append(next_vertex)
		if len(next_vertex.parents) > 1:
			if next_vertex not in disjunctions:
				disjunctions.append(next_vertex)
		for neighbor in next_vertex.children + next_vertex.parents:
			if neighbor not in visited:
				queue.append(neighbor)

	debug_msg = None
	known_values = {}
	known_values[start_vertex] = [], [(frozenset(),[])] # start_vertex is provable from the set of no axioms
	queue = start_vertex.children + start_vertex.parents
	while len(queue) != 0:
		next_vertex = queue.pop()
		if print_debug_info:
			print("solve_causal_query DEBUG: Processing vertex '{}'.".format(next_vertex))

		if next_vertex in known_values:
			provably_false, provably_true = known_values[next_vertex]
		else:
			provably_true = []
			provably_false = []
		provably_true_changed = False
		provably_false_changed = False

		# check if any of its parents are set to true
		for parent in next_vertex.parents:
			if parent in known_values:
				for hypotheses, _ in known_values[parent][1]:
					if print_debug_info:
						debug_msg = "solve_causal_query DEBUG: '{}' is provably true since its parent '{}' is true given {}.".format(next_vertex, parent, hypotheses)
					provably_true, provably_true_changed = add_hypothesis_set(provably_true, hypotheses, parent, provably_true_changed, debug_msg)

		# check if all parents are false
		if len(next_vertex.parents) != 0:
			combinations = itertools.product(*[(known_values[parent][0] if parent in known_values else []) for parent in next_vertex.parents])
			new_provably_false = reduce_hypothesis_set([frozenset().union(*[hypotheses for (hypotheses, _) in combination]) for combination in combinations])
			for hypotheses, _ in new_provably_false:
				if print_debug_info:
					debug_msg = "solve_causal_query DEBUG: '{}' is provably false since all parents are false given {}.".format(next_vertex, hypotheses)
				provably_false, provably_false_changed = add_hypothesis_set(provably_false, hypotheses, next_vertex.parents, provably_false_changed, debug_msg)

		for child in next_vertex.children:
			if child in known_values:
				# check if any child is true and all other parents are false
				hypotheses_lists = [known_values[child][1]] + [(known_values[parent][0] if parent in known_values else []) for parent in child.parents if parent != next_vertex]
				combinations = itertools.product(*hypotheses_lists)
				new_provably_true = reduce_hypothesis_set([frozenset().union(*[hypotheses for (hypotheses, _) in combination]) for combination in combinations])
				for hypotheses, _ in new_provably_true:
					if print_debug_info:
						debug_msg = "solve_causal_query DEBUG: '{}' is provably true since its child '{}' is true and all other parents are false given {}.".format(next_vertex, child, hypotheses)
					provably_true, provably_true_changed = add_hypothesis_set(provably_true, hypotheses, child, provably_true_changed, debug_msg)

				# check if any child is set to false
				for hypotheses, _ in known_values[child][0]:
					if print_debug_info:
						debug_msg = "solve_causal_query DEBUG: '{}' is provably false since its child '{}' is false given {}.".format(next_vertex, child, hypotheses)
					provably_false, provably_false_changed = add_hypothesis_set(provably_false, hypotheses, child, provably_false_changed, debug_msg)

		# check if we can apply proof-by-cases
		if provably_true_changed:
			for disjunction in disjunctions:
				if disjunction not in known_values:
					continue
				hypotheses_lists = [[(hypotheses, proof_edges) for (hypotheses, proof_edges) in provably_true if case in hypotheses] for case in disjunction.parents]
				hypotheses_lists.append(known_values[disjunction][1])
				for combination in itertools.product(*hypotheses_lists):
					new_combination = []
					for i in range(len(disjunction.parents)):
						new_combination.append((set(combination[i][0]), combination[i][1]))
						new_combination[i][0].remove(disjunction.parents[i])
					new_hypotheses = frozenset().union(*[hypotheses for (hypotheses, proof_edges) in new_combination + [combination[-1]]])
					if print_debug_info:
						debug_msg = "solve_causal_query DEBUG: '{}' can be proved true by cases {} given {}.".format(next_vertex, disjunction.parents, new_hypotheses)
					provably_true, provably_true_changed = add_hypothesis_set(provably_true, new_hypotheses, tuple([proof_edges[:] for (hypotheses, proof_edges) in new_combination] + [disjunction]), provably_true_changed, debug_msg)
		if provably_false_changed:
			for disjunction in disjunctions:
				if disjunction not in known_values:
					continue
				hypotheses_lists = [[(hypotheses, proof_edges) for (hypotheses, proof_edges) in provably_false if case in hypotheses] for case in disjunction.parents]
				hypotheses_lists.append(known_values[disjunction][1])
				for combination in itertools.product(*hypotheses_lists):
					new_combination = []
					for i in range(len(disjunction.parents)):
						new_combination.append((set(combination[i][0]), combination[i][1]))
						new_combination[i][0].remove(disjunction.parents[i])
					new_hypotheses = frozenset().union(*[hypotheses for (hypotheses, proof_edges) in new_combination + [combination[-1]]])
					if print_debug_info:
						debug_msg = "solve_causal_query DEBUG: '{}' can be proved false by cases {} given {}.".format(next_vertex, disjunction.parents, new_hypotheses)
					provably_false, provably_false_changed = add_hypothesis_set(provably_false, new_hypotheses, tuple([proof_edges[:] for (hypotheses, proof_edges) in new_combination] + [disjunction]), provably_false_changed, debug_msg)

		known_values[next_vertex] = provably_false, provably_true

		if provably_true_changed or provably_false_changed:
			for neighbor in next_vertex.children + next_vertex.parents:
				queue.append(neighbor)

		if len(queue) == 0:
			# consider proof-by-cases
			to_enqueue = []
			for disjunction in disjunctions:
				if disjunction in known_values and len(known_values[disjunction][1]) != 0:
					for parent in disjunction.parents:
						if parent not in known_values:
							known_values[parent] = [], [(frozenset([parent]), [])]
							for neighbor in parent.children + parent.parents:
								if neighbor not in to_enqueue:
									to_enqueue.append(neighbor)
						else:
							parent_provably_false, parent_provably_true = known_values[parent]
							parent_provably_true, updated = add_hypothesis_set(parent_provably_true, frozenset([parent]), None, False, None)
							if updated:
								known_values[parent] = parent_provably_false, parent_provably_true
								for neighbor in parent.children + parent.parents:
									if neighbor not in to_enqueue:
										to_enqueue.append(neighbor)
			queue.extend(to_enqueue)

	# enumerate all proofs
	if target_vertex in known_values and any([len(hypotheses) == 0 for (hypotheses, _) in known_values[target_vertex][1]]):
		proofs = enumerate_hyperpaths(known_values, start_vertex, target_vertex, frozenset())
		return known_values, [proof + [target_vertex] for proof in proofs if target_vertex not in proof]
	else:
		# perform a DFS to prove all provable events
		inv_proof_graph = {}
		for vertex, (provably_false, provably_true) in known_values.items():
			for (hypotheses, proof_edges) in provably_true:
				for proof_edge in proof_edges:
					if type(proof_edge) == list:
						proof_edge = frozenset(proof_edge)
					elif type(proof_edge) == tuple:
						new_tuple = []
						for case in proof_edge[:-1]:
							new_tuple.append(frozenset([x for x in case if type(x) == Node]))
						new_tuple.append(proof_edge[-1])
						proof_edge = tuple(new_tuple)
					if proof_edge not in inv_proof_graph:
						provably_true = []
						provably_false = []
					else:
						provably_false, provably_true = inv_proof_graph[proof_edge]
					provably_true, _ = add_hypothesis_set(provably_true, hypotheses, vertex, False, None)
					inv_proof_graph[proof_edge] = provably_false, provably_true
			for (hypotheses, proof_edges) in provably_false:
				for proof_edge in proof_edges:
					if type(proof_edge) == list:
						proof_edge = frozenset(proof_edge)
					elif type(proof_edge) == tuple:
						new_tuple = []
						for case in proof_edge[:-1]:
							new_tuple.append(tuple([x for x in case if type(x) == Node]))
						new_tuple.append(proof_edge[-1])
						proof_edge = tuple(new_tuple)
					if proof_edge not in inv_proof_graph:
						provably_true = []
						provably_false = []
					else:
						provably_false, provably_true = inv_proof_graph[proof_edge]
					provably_false, _ = add_hypothesis_set(provably_false, hypotheses, vertex, False, None)
					inv_proof_graph[proof_edge] = provably_false, provably_true

		proof = []
		queue.append(start_vertex)
		while len(queue) != 0:
			next_vertex = queue.pop()
			if next_vertex in proof:
				continue
			proof.append(next_vertex)

			for proof_edge, (provably_false, provably_true) in inv_proof_graph.items():
				# check if any conjunctions of vertices are satisfied
				if type(proof_edge) == frozenset:
					is_edge_provable = True
					for element in proof_edge:
						if element not in proof:
							is_edge_provable = False
							break
					if is_edge_provable:
						for hypotheses, vertices in provably_true:
							if hypotheses != frozenset():
								continue
							for vertex in vertices:
								if vertex not in proof:
									queue.append(vertex)
				# check if any disjunctions of vertices are satisfied
				elif type(proof_edge) == tuple and proof_edge[-1] in proof:
					all_cases_proved = True
					for i in range(len(proof_edge[-1].parents)):
						if not any([edge in proof for edge in proof_edge[i]]):
							all_cases_proved = False
							break
					if all_cases_proved:
						for hypotheses, vertices in provably_true:
							if hypotheses != frozenset():
								continue
							for vertex in vertices:
								if vertex not in proof:
									queue.append(vertex)

			if next_vertex in inv_proof_graph:
				provably_false, provably_true = inv_proof_graph[next_vertex]
				for hypotheses, vertices in provably_true:
					if hypotheses != frozenset():
						continue
					for vertex in vertices:
						if vertex not in proof:
							queue.append(vertex)
		return known_values, [proof]

def generate_example(num_vertices, max_num_parents, max_cyclic_edges, intervene):
	vertices = generate_graph(num_vertices, max_num_parents, max_cyclic_edges)

	available_actors = ["square", "triangle", "circle", "star", "stone", "lightbulb", "water", "smoke", "sand"]
	available_colors = ["red", "blue", "green", "orange", "purple", "yellow", "white", "black"]

	for vertex in vertices:
		vertex.color = choice(available_colors)
		vertex.actor = choice(available_actors)
		available_actors.remove(vertex.actor)

	# convert the graph into a context
	context = []
	for vertex in vertices:
		causes = [parent.to_clause("be") for parent in vertex.parents]
		if len(causes) == 0:
			continue
		elif len(causes) == 1:
			sentence = vertex.to_clause("turn") + " only when " + causes[0] + "."
		elif len(causes) == 2:
			sentence = vertex.to_clause("turn") + " only when " + causes[0] + "."
			sentence = vertex.to_clause("turn") + " only when " + causes[0] + " or " + causes[1] + "."
		else:
			sentence = vertex.to_clause("turn") + " only when " + ", ".join(causes[:-1]) + ", or " + causes[-1] + "."
		context.append(capitalize(sentence))

	# generate a question
	events = sample(vertices, 2)
	query_state = True #choice([True, False])
	if intervene:
		query = "Suppose we force " + events[0].to_clause("turn", finite=False, negate=not query_state) + ", without forcing anything else. " + capitalize(events[1].to_clause("be", invert=True, adverb="necessarily")) + "?"

		# remove the parents of the event that we caused
		for parent in list(events[0].parents):
			parent.children.remove(events[0])
			events[0].parents.remove(parent)
	else:
		query = "Suppose " + events[0].to_clause("be", negate=not query_state) + ". " + capitalize(events[1].to_clause("be", invert=True, adverb="necessarily")) + "?"

	# compute the solution to the query
	if print_debug_info:
		print("\ngenerate_example DEBUG: Calling `solve_causal_query` with example {} and query '{}'".format(context, query))
	known_values, proof_paths = solve_causal_query(events[0], query_state, events[1])
	are_unique(proof_paths)
	proofs = []
	for i in range(len(proof_paths)):
		proof = []
		for j in range(len(proof_paths[i])):
			verb = ("be" if j == 0 else "must be")
			if type(proof_paths[i][j]) == list:
				sentence = capitalize(" and ".join([vertex.to_clause(verb) for vertex in proof_paths[i][j]])) + "."
			elif type(proof_paths[i][j]) == tuple:
				sentence = "Either " + " or ".join([vertex.to_clause(verb) for vertex in proof_paths[i][j]]) + "."
			else:
				sentence = capitalize(proof_paths[i][j].to_clause(verb)) + "."
			proof.append(sentence)
		if proof_paths[i][-1] != events[1]:
			# this is a "proof" that the target event is not provable
			proof.append("But nothing else is necessarily true.")
			proof.append("The " + events[1].actor + " is not necessarily " + events[1].color + ".")
		proofs.append(proof)

	if events[1] in known_values and any([len(hypotheses) == 0 for (hypotheses, _) in known_values[events[1]][1]]):
		answer = "Yes"
	else:
		answer = "No"

	return context, query, proofs, answer

def parse_log(log):
	trial = 0
	resume_position = 0
	last_question = ""
	expected_answers = []
	predicted_answers = []
	while True:
		# look for the next line beginning with 'Predicted answer:'
		line = log.readline()
		if not line:
			break # found the end of the file
		elif line.startswith('(Example '):
			last_question = line[(line.index('Question: ') + len('Question: ')):]
			continue
		elif not line.startswith('Predicted answer:'):
			last_question += line
			continue

		# read the predicted answer
		expected_answer = None
		predicted_answer = line[len('Predicted answer:'):]
		while True:
			line = log.readline()
			if not line:
				break # found the end of the file
			elif line.startswith('Expected answer: '):
				expected_answer = line[len('Expected answer: '):]
				break
			predicted_answer += line

		# read the expected answer
		mean = None
		found_summary = False
		while expected_answer is not None:
			line = log.readline()
			if not line:
				break # found the end of the file
			elif line.startswith('Logprobs: '):
				# read the summary statistics
				log.readline() # consume the empty line separating each example
				trial += 1
				resume_position = log.tell()
				found_summary = True
				break
			expected_answer += line

		if not found_summary:
			break
		expected_answers.append(expected_answer)
		predicted_answers.append(predicted_answer)

	return (trial, resume_position, expected_answers, predicted_answers)

def print_output(str, log):
	log.write(str + '\n')
	print(str)

if __name__ == "__main__":
	seed(456130212)
	use_chain_of_thought = True
	num_few_shot_examples = 0
	num_trials = 100
	model = "gpt3"
	model_size = "text-davinci-003"
	intervention = True

	log_file = model + "_" + model_size.lower().replace('-', '') + "_" + str(num_few_shot_examples) + "shot"
	if use_chain_of_thought:
		if num_few_shot_examples == 0:
			log_file += "_stepbystep"
		else:
			log_file += "_CoT"
	if intervention:
		log_file += "_intervention"
	log_file += ".log"

	log = open(log_file, "a+")
	log.seek(0)
	(resume_trial, truncate_pos, _, _) = parse_log(log)
	if truncate_pos != 0:
		print("Resuming existing experiment at trial {}".format(resume_trial + 1))

	if model == "gpt3":
		gpt_api_key = getpass.getpass(prompt='Enter OpenAI API Key:')
	trial = 0
	while trial < num_trials:
		prompt = ""
		for i in range(num_few_shot_examples):
			expected_answer = choice(["Yes", "No"])
			while True:
				context, query, proofs, answer = generate_example(5, 3, 0, intervene=intervention)
				if len(proofs) == 1 and answer == expected_answer:
					break
			if use_chain_of_thought:
				proof_text = ((" ".join(proofs[0]) + " ") if len(proofs) != 0 else "")
				prompt += "Question: " + " ".join(context) + "\n" + query + "\nExplanation: " + proof_text + "\nAnswer: " + answer + "\n\n"
			else:
				prompt += "Question: " + " ".join(context) + "\n" + query + "\nAnswer: " + answer + "\n\n"

		expected_answer = choice(["Yes", "No"])
		while True:
			context, query, proofs, answer = generate_example(5, 3, 0, intervene=intervention)
			if len(proofs) == 1 and answer == expected_answer:
				break
			#elif len(proofs) != 1:
			#	print("DEBUG: This question has multiple proofs.")
			#	print("DEBUG: " + " ".join(context) + "\n" + query)
			#	for i in range(len(proofs)):
			#		print("DEBUG[{}]: {}\n".format(i, " ".join(proofs[i])))
		prompt += "Question: " + " ".join(context) + "\n" + query
		if use_chain_of_thought:
			if num_few_shot_examples == 0:
				prompt += "\nLet's think step-by-step, then answer Yes or No:"
			else:
				prompt += "\nExplanation:"
		else:
			prompt += "\nAnswer:"

		trial += 1
		if trial <= resume_trial:
			continue
		log.write("(Example {}) ".format(trial))
		print_output(prompt, log)

		if model == "gpt3":
			prediction, logprobs = gpt3.predict(gpt_api_key, "text-davinci-003", prompt)
		elif model == "dummy":
			prediction, logprobs = "", []
		print_output("Predicted answer:" + prediction, log)
		if use_chain_of_thought:
			proof_text = ((" ".join(proofs[0]) + " ") if len(proofs) != 0 else "")
			print_output("Expected answer: " + proof_text + answer + "", log)
		else:
			print_output("Expected answer: " + answer + "", log)
		print_output("Logprobs: " + json.dumps(logprobs) + "\n", log)
