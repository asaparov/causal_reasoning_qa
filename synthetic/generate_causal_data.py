import numpy as np
from random import seed, choice, sample, shuffle, randrange, random
from fol import *

class Node(object):
	def __init__(self, id):
		self.id = id
		self.children = []
		self.parents = []

	def __str__(self):
		return 'n(' + str(self.id) + ')'

	def __repr__(self):
		return 'n(' + str(self.id) + ')'

	def name(self):
		return "event" + str(self.id)

def generate_graph(num_vertices, num_roots, id_offset=0, alpha=3.0):
	vertices = []
	for i in range(num_vertices):
		vertices.append(Node(i))

	# sample a random DAG
	for i in range(num_roots, num_vertices):
		# sample the number of parent vertices
		num_parents = np.random.zipf(alpha)
		num_parents = min(num_parents, i)

		for parent_id in sample(range(i), num_parents):
			vertices[parent_id].children.append(vertices[i])
			vertices[i].parents.append(vertices[parent_id])

	# make sure each root has at least one child
	for root in vertices[:num_roots]:
		if len(root.children) == 0:
			node = choice(vertices[num_roots:])
			root.children.append(node)
			node.parents.append(root)

	# remove any correlation between graph topology and vertex IDs by shuffling the vertices
	shuffle(vertices)
	for i in range(len(vertices)):
		vertices[i].id = id_offset + i
	return vertices

def get_ancestors(node):
	queue = [node]
	descendants = set()
	while len(queue) != 0:
		node = queue.pop()
		if node in descendants:
			continue
		descendants.add(node)
		for parent in node.parents:
			queue.append(parent)
	return descendants

def get_descendants(node):
	queue = [node]
	descendants = set()
	while len(queue) != 0:
		node = queue.pop()
		if node in descendants:
			continue
		descendants.add(node)
		for child in node.children:
			queue.append(child)
	return descendants

def generate_graph_and_scenarios(num_vertices, num_scenarios, generate_cause_edges, generate_non_causal_relations, generate_negative_cause_edges, generate_non_occuring_events, generate_counterfactuals, generate_negative_counterfactuals, id_offset=0, mean_scenario_length=4):
	# create a graph where the number of roots is num_vertices / 64, on average
	num_roots = np.random.geometric(64 / num_vertices)
	num_roots = min(num_roots, num_vertices / 16) # make sure there aren't too many roots
	causal_graph = generate_graph(num_vertices, num_roots, id_offset)

	causal_graph_lfs, negative_cause_lfs = [], []
	if generate_cause_edges:
		# print the causal graph
		stack = causal_graph[:num_roots]
		visited = set()
		while len(stack) != 0:
			node = stack.pop()
			if node in visited:
				continue
			visited.add(node)
			for child in node.children:
				lf = FOLFuncApplication("cause", [FOLConstant(node.name()), FOLConstant(child.name())])
				causal_graph_lfs.append(lf)
				stack.append(child)

	# sample a bunch of negative edges
	negative_edges = set()
	while len(negative_edges) < num_vertices / 2:
		# since the graph is a DAG, any edge that creates a cycle is negative
		src = choice(causal_graph)
		ancestors = get_ancestors(src)
		ancestors.remove(src)
		if len(ancestors) == 0:
			continue
		dst = choice(list(ancestors))
		negative_edges.add((src, dst))
	if generate_negative_cause_edges:
		for (src, dst) in negative_edges:
			lf = FOLNot(FOLFuncApplication("cause", [FOLConstant(src.name()), FOLConstant(dst.name())]))
			negative_cause_lfs.append(lf)

	# sample a number of scenario instances (each scenario is a description of
	# a set of events, where some of the events occur causally, some occur non-
	# causally, and some do not occur; the description contains other relations
	# as well, such as temporal ordering)
	scenarios = []
	while len(scenarios) < num_scenarios:
		start_event = choice(causal_graph)
		if len(start_event.children) == 0 or len(start_event.parents) == 0:
			continue

		# sample a number of causally-disconnected sets of events
		num_event_clusters = 1 + np.random.geometric(0.25)
		num_event_clusters = min(num_event_clusters, num_roots)
		event_clusters = [([root], get_descendants(root)) for root in sample(causal_graph[:num_roots], num_event_clusters)]
		for event_cluster, cluster_descendants in event_clusters:
			# sample a chain of events caused by the start_event
			current_event = event_cluster[0]
			while True:
				valid_children = [child for child in current_event.children if all([child not in descendants for _, descendants in event_clusters if descendants != cluster_descendants])]
				if len(valid_children) == 0:
					break
				current_event = choice(valid_children)
				event_cluster.append(current_event)
				if random() < 1.0 / mean_scenario_length:
					break

		# select a subset of event clusters that do not occur
		num_non_occuring_events = np.random.binomial(num_event_clusters - 1, 0.2)
		non_occuring_events = sample(event_clusters, num_non_occuring_events)
		occuring_events = [cluster for cluster in event_clusters if cluster not in non_occuring_events]

		# of the occuring events clusters, sample a temporal ordering (for the `precede` relation)
		temporal_ordering = occuring_events
		shuffle(temporal_ordering)

		scenario_lfs = []
		for event_cluster in event_clusters:
			event_chain, _ = event_cluster
			if event_cluster in occuring_events:
				# describe events that occur
				for event in event_chain:
					lf = FOLFuncApplication("occur", [FOLConstant(event.name())])
					scenario_lfs.append(lf)

				if generate_non_causal_relations:
					# describe the temporal ordering of a subset of events
					temporal_edges = set()
					num_temporal_sources = np.random.binomial(len(event_chain), 0.5)
					temporal_sources = sample(event_chain, num_temporal_sources)
					for src in temporal_sources:
						# sample another event either in any cluster
						sampled_cluster = choice(occuring_events)
						sampled_events, _ = sampled_cluster
						if sampled_events == [src]:
							continue
						sampled_event = choice([event for event in sampled_events if event != src])
						# determine which event happened first
						if sampled_events == event_chain:
							if event_chain.index(sampled_event) < event_chain.index(src):
								first_event, second_event = sampled_event, src
							else:
								first_event, second_event = src, sampled_event
						else:
							if temporal_ordering.index(sampled_cluster) < temporal_ordering.index(event_cluster):
								first_event, second_event = sampled_event, src
							else:
								first_event, second_event = src, sampled_event
						temporal_edges.add((first_event, second_event))
					for (src, dst) in temporal_edges:
						lf = FOLFuncApplication("precede", [FOLConstant(src.name()), FOLConstant(dst.name())])
						scenario_lfs.append(lf)

					# describe events that are co-located
					num_colocated_sources = np.random.binomial(len(event_chain), 0.4)
					colocated_sources = sample(event_chain, num_colocated_sources)
					for src in colocated_sources:
						# sample another event either in any cluster
						sampled_cluster = choice(occuring_events)
						sampled_events, _ = sampled_cluster
						dst = choice(sampled_events)
						# determine if the events are colocated
						if sampled_events == event_chain:
							lf = FOLFuncApplication("colocate", [FOLConstant(src.name()), FOLConstant(dst.name())])
							scenario_lfs.append(lf)
						else:
							lf = FOLNot(FOLFuncApplication("colocate", [FOLConstant(src.name()), FOLConstant(dst.name())]))
							scenario_lfs.append(lf)

				# generate counterfactual examples
				counterfactual_edges = set()
				num_counterfactual_examples = np.random.binomial(len(event_chain), 0.4)
				counterfactual_sources = sample(event_chain, num_counterfactual_examples)
				for src in counterfactual_sources:
					# sample another event in this cluster
					if event_chain == [src]:
						continue
					dst = choice([event for event in event_chain if event != src])
					if event_chain.index(src) < event_chain.index(dst):
						if generate_counterfactuals:
							lf = FOLFuncApplication("counterfactual", [FOLConstant(src.name()), FOLConstant(dst.name())])
							scenario_lfs.append(lf)
					else:
						if generate_negative_counterfactuals:
							lf = FOLNot(FOLFuncApplication("counterfactual", [FOLConstant(src.name()), FOLConstant(dst.name())]))
							scenario_lfs.append(lf)

				# generate negative counterfactual examples
				num_counterfactual_examples = np.random.binomial(len(event_chain), 0.2)
				counterfactual_sources = sample(event_chain, num_counterfactual_examples)
				for src in counterfactual_sources:
					# sample another event either in any cluster
					sampled_cluster = choice(occuring_events)
					sampled_events, _ = sampled_cluster
					dst = choice(sampled_events)
					if sampled_events == event_chain and event_chain.index(src) < event_chain.index(dst):
						if generate_counterfactuals:
							lf = FOLFuncApplication("counterfactual", [FOLConstant(src.name()), FOLConstant(dst.name())])
							scenario_lfs.append(lf)
					else:
						if generate_negative_counterfactuals:
							lf = FOLNot(FOLFuncApplication("counterfactual", [FOLConstant(src.name()), FOLConstant(dst.name())]))
							scenario_lfs.append(lf)

			elif generate_non_occuring_events:
				# describe events that do not occur
				for event in event_chain:
					lf = FOLNot(FOLFuncApplication("occur", [FOLConstant(event.name())]))
					scenario_lfs.append(lf)

		scenarios.append(scenario_lfs)

	return causal_graph, causal_graph_lfs, negative_cause_lfs, scenarios

def try_capitalize(sentence, capitalize):
	if not capitalize:
		return sentence
	return sentence[0].upper() + sentence[1:]

def logical_form_to_clause(lf, use_synonym_for_cause, use_synonym_for_happen, use_synonym_for_precede, use_synonym_for_colocate, capitalize):
	if type(lf) == FOLNot and type(lf.operand) == FOLFuncApplication:
		if lf.operand.function == "cause":
			if use_synonym_for_cause and choice([True, False]):
				clause = lf.operand.args[0].constant + " does not lead to " + lf.operand.args[1].constant
			else:
				clause = lf.operand.args[0].constant + " does not cause " + lf.operand.args[1].constant
		elif lf.operand.function == "occur":
			if use_synonym_for_happen and choice([True, False]):
				clause = lf.operand.args[0].constant + " did not occur"
			else:
				clause = lf.operand.args[0].constant + " did not happen"
		elif lf.operand.function == "precede":
			if use_synonym_for_precede and choice([True, False]):
				clause = lf.operand.args[0].constant + " did not happen before " + lf.operand.args[1].constant
			else:
				clause = lf.operand.args[0].constant + " did not precede " + lf.operand.args[1].constant
		elif lf.operand.function == "colocate":
			if use_synonym_for_colocate and choice([True, False]):
				clause = lf.operand.args[0].constant + " and " + lf.operand.args[1].constant + " were not colocated"
			else:
				clause = lf.operand.args[0].constant + " and " + lf.operand.args[1].constant + " did not happen in the same place"
		elif lf.operand.function == "counterfactual":
			if use_synonym_for_happen and choice([True, False]):
				clause = "if " + lf.operand.args[0].constant + " did not occur, and " + lf.operand.args[1].constant + " has no other causes, would " + lf.operand.args[1].constant + " occur? " + try_capitalize("yes", capitalize)
			else:
				clause = "if " + lf.operand.args[0].constant + " did not happen, and " + lf.operand.args[1].constant + " has no other causes, would " + lf.operand.args[1].constant + " happen? " + try_capitalize("yes", capitalize)
		else:
			raise ValueError("Unsupported logical form.")
	elif type(lf) == FOLFuncApplication:
		if lf.function == "cause":
			if use_synonym_for_cause and choice([True, False]):
				clause = lf.args[0].constant + " leads to " + lf.args[1].constant
			else:
				clause = lf.args[0].constant + " causes " + lf.args[1].constant
		elif lf.function == "occur":
			if use_synonym_for_happen and choice([True, False]):
				clause = lf.args[0].constant + " occured"
			else:
				clause = lf.args[0].constant + " happened"
		elif lf.function == "precede":
			if use_synonym_for_precede and choice([True, False]):
				clause = lf.args[0].constant + " happened before " + lf.args[1].constant
			else:
				clause = lf.args[0].constant + " preceded " + lf.args[1].constant
		elif lf.function == "colocate":
			if use_synonym_for_colocate and choice([True, False]):
				clause = lf.args[0].constant + " and " + lf.args[1].constant + " were colocated"
			else:
				clause = lf.args[0].constant + " and " + lf.args[1].constant + " happened in the same place"
		elif lf.function == "counterfactual":
			if use_synonym_for_happen and choice([True, False]):
				clause = "if " + lf.args[0].constant + " did not occur, and " + lf.args[1].constant + " has no other causes, would " + lf.args[1].constant + " occur? " + try_capitalize("no", capitalize)
			else:
				clause = "if " + lf.args[0].constant + " did not happen, and " + lf.args[1].constant + " has no other causes, would " + lf.args[1].constant + " happen? " + try_capitalize("no", capitalize)
		else:
			raise ValueError("Unsupported logical form.")
	else:
		raise ValueError("Unsupported logical form.")
	return try_capitalize(clause, capitalize) + '.'

if __name__ == "__main__":
	import argparse
	def parse_bool_arg(v):
		if isinstance(v, bool):
			return v
		elif v.lower() in ('yes', 'true', 'y', 't', '1'):
			return True
		elif v.lower() in ('no', 'false', 'n', 'f', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')

	parser = argparse.ArgumentParser()
	parser.add_argument("--num-vertices", type=int, required=True)
	parser.add_argument("--num-scenarios", type=int, required=True)
	parser.add_argument("--logical-form", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--print-causal-graph", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--generate-non-causal-relations", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--generate-negative-cause-edges", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--generate-non-occuring-events", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--generate-counterfactuals", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--generate-negative-counterfactuals", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--use-synonym-for-cause", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--use-synonym-for-happen", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--use-synonym-for-precede", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--use-synonym-for-colocate", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--mean-scenario-length", type=int, default=4)
	parser.add_argument("--dont-capitalize", action='store_true')
	parser.add_argument("--seed", type=int, default=62471893)
	args = parser.parse_args()

	seed(args.seed)
	np.random.seed(args.seed)
	causal_graph, causal_graph_lfs, negative_cause_lfs, scenarios = generate_graph_and_scenarios(
		num_vertices=args.num_vertices,
		num_scenarios=args.num_scenarios,
		generate_cause_edges=args.print_causal_graph,
        generate_non_causal_relations=args.generate_non_causal_relations,
		generate_negative_cause_edges=args.generate_negative_cause_edges,
		generate_non_occuring_events=args.generate_non_occuring_events,
		generate_counterfactuals=args.generate_counterfactuals,
		generate_negative_counterfactuals=args.generate_negative_counterfactuals,
		mean_scenario_length=args.mean_scenario_length)

	from sys import stdout
	first = True
	for lf in causal_graph_lfs:
		if args.logical_form:
			sentence = fol_to_tptp(lf) + '.'
		else:
			sentence = logical_form_to_clause(lf,
				use_synonym_for_cause=args.use_synonym_for_cause,
				use_synonym_for_happen=args.use_synonym_for_happen,
				use_synonym_for_precede=args.use_synonym_for_precede,
				use_synonym_for_colocate=args.use_synonym_for_colocate,
				capitalize=not args.dont_capitalize)
		if not first:
			stdout.write(' ')
		first = False
		stdout.write(sentence)

	if len(negative_cause_lfs) != 0:
		stdout.write('\n\n')
		first = True
		for lf in negative_cause_lfs:
			if args.logical_form:
				sentence = fol_to_tptp(lf) + '.'
			else:
				sentence = logical_form_to_clause(lf,
					use_synonym_for_cause=args.use_synonym_for_cause,
					use_synonym_for_happen=args.use_synonym_for_happen,
					use_synonym_for_precede=args.use_synonym_for_precede,
					use_synonym_for_colocate=args.use_synonym_for_colocate,
					capitalize=not args.dont_capitalize)
			if not first:
				stdout.write(' ')
			first = False
			stdout.write(sentence)

	for scenario in scenarios:
		first = True
		stdout.write('\n\n') # optionally, add some text to mark the beginning of a new scenario
		for lf in scenario:
			if args.logical_form:
				sentence = fol_to_tptp(lf) + '.'
			else:
				sentence = logical_form_to_clause(lf,
					use_synonym_for_cause=args.use_synonym_for_cause,
					use_synonym_for_happen=args.use_synonym_for_happen,
					use_synonym_for_precede=args.use_synonym_for_precede,
					use_synonym_for_colocate=args.use_synonym_for_colocate,
					capitalize=not args.dont_capitalize)
			if not first:
				stdout.write(' ')
			first = False
			stdout.write(sentence)

	stdout.write('\n')
