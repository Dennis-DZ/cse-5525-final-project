from src.data_loader import EvaluationOutput
import zss # type: ignore
import numbers
import json
import jsonschema
import logging

class Tree:
	def __init__(self, label, is_value=False):
		self.label = label
		self.is_value = is_value
		self.children = []

	@staticmethod
	def get_children(tree):
		return tree.children
	
	@staticmethod
	def get_label(node):
		return node.label

	def size(self):
		size = 1
		for child in self.children:
			size += child.size()
		return size
	
	def get_paths(self, include_values=False):
		if len(self.children) == 0:
			if not self.is_value:
				return set([str(self.label) + "/"])
			elif include_values:
				return set([str(self.label)])
			else:
				return set([""])

		paths = set()
		for child in self.children:
			child_paths = child.get_paths(include_values)
			paths.update([f"{self.label}/{path}" for path in child_paths])

		return paths

def build_tree(data, label, schema):
	tree = Tree(label)

	if isinstance(data, dict):
		properties_schema = schema.get("properties") if schema else None

		for key in sorted(data.keys()):
			child_schema = properties_schema.get(key) if properties_schema else None

			if child_schema and child_schema.get("type") == "string" and "enum" not in child_schema:
				# Ignore non-enum strings
				continue

			sub_tree = build_tree(
				data=data[key],
				label=key,
				schema=child_schema
			)

			tree.children.append(sub_tree)

	elif isinstance(data, list):
		sub_trees = []
		item_schema = schema.get("items") if schema else None

		if item_schema and item_schema.get("type") == "string" and "enum" not in item_schema:
			# Ignore non-enum strings
			return tree

		for i, item in enumerate(data):
			sub_tree = build_tree(
				data=item,
				label=None,
				schema=item_schema
			)

			sort_key = json.dumps(item, sort_keys=True)
			sub_trees.append((sort_key, sub_tree))

		if schema and schema.get("ordered") is False:
			sub_trees.sort(key=lambda x: x[0])

		for i, (sort_key, sub_tree) in enumerate(sub_trees):
			sub_tree.label = f"item_{i}"
			tree.children.append(sub_tree)

	else:
		if schema and schema.get("type") == "string" and "enum" not in schema:
			# Ignore non-enum strings
			return tree
		else:
			tree.children.append(Tree(data, is_value=True))

	return tree

def label_distance(true_label, pred_label):
	if true_label == pred_label:
		return 0
	
	if not isinstance(true_label, numbers.Number) \
			or not isinstance(pred_label, numbers.Number):
		return 1
	
	denominator = max(abs(true_label), abs(pred_label), 1e-6)
	cost = abs(true_label - pred_label) / denominator
	return min(cost, 1)

def tree_edit_distance(true_tree, pred_tree):
	distance = zss.simple_distance(
		true_tree,
		pred_tree,
		Tree.get_children,
		Tree.get_label,
		label_distance
	)

	max_distance = true_tree.size() + pred_tree.size()
	if max_distance == 0:
		return 1, 0
	
	similarity = 1 - (distance / max_distance)
	return similarity, distance

def calculate_f1(true_set, pred_set):
	true_positives = len(true_set & pred_set)
	false_positives = len(pred_set - true_set)
	false_negatives = len(true_set - pred_set)

	if true_positives + false_positives == 0:
		precision = 0
	else:
		precision = true_positives / (true_positives + false_positives)

	if true_positives + false_negatives == 0:
		recall = 0
	else:
		recall = true_positives / (true_positives + false_negatives)

	if 2 * true_positives + false_positives + false_negatives == 0:
		f1 = 0
	else:
		f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

	return f1, precision, recall

def key_and_item_f1(true_tree, pred_tree):
	true_keys = true_tree.get_paths(include_values=False)
	pred_keys = pred_tree.get_paths(include_values=False)
	key_f1, key_precision, key_recall = calculate_f1(true_keys, pred_keys)

	true_items = true_tree.get_paths(include_values=True)
	pred_items = pred_tree.get_paths(include_values=True)
	item_f1, item_precision, item_recall = calculate_f1(true_items, pred_items)

	return {
		"key": {
			"f1": key_f1,
			"precision": key_precision,
			"recall": key_recall
		},
		"item": {
			"f1": item_f1,
			"precision": item_precision,
			"recall": item_recall
		}
	}

def calculate_eval_metrics(true_spec_string, pred_spec_string, schema) -> EvaluationOutput:
	logger = logging.getLogger("__main__")
	scores = EvaluationOutput()

	try:
		true_spec = json.loads(true_spec_string)
	except json.JSONDecodeError:
		logger.error(f"True spec isn't valid JSON\n{true_spec_string}")
		return scores

	try:
		pred_spec = json.loads(pred_spec_string)
		scores.valid_json = True
	except json.JSONDecodeError:
		scores.valid_json = False
		return scores

	try:
		jsonschema.validate(true_spec, schema)
	except jsonschema.ValidationError as error:
		logger.error(f"True spec doesn't match schema: {error.message}\n{error.instance}")

	try:
		jsonschema.validate(pred_spec, schema)
		scores.schema_match = True
	except jsonschema.ValidationError as error:
		scores.schema_match = False
		scores.schema_error_message = error.message
		print(error.schema_path)
		scores.schema_error_path = ".".join(str(x) for x in error.schema_path)

	true_tree = build_tree(true_spec, "root", schema)
	pred_tree = build_tree(pred_spec, "root", schema)

	f1_scores = key_and_item_f1(true_tree, pred_tree)
	similarity, distance = tree_edit_distance(true_tree, pred_tree)

	scores.key_f1 = f1_scores["key"]["f1"]
	scores.key_precision = f1_scores["key"]["precision"]
	scores.key_recall = f1_scores["key"]["recall"]

	scores.item_f1 = f1_scores["item"]["f1"]
	scores.item_precision = f1_scores["item"]["precision"]
	scores.item_recall = f1_scores["item"]["recall"]

	scores.tree_edit_similarity = similarity
	scores.tree_edit_distance = distance

	return scores

def main():
	import jsonref # type: ignore
	from dataclasses import asdict

	with open("data/simulation_schema.json", "r", encoding="utf-8") as file:
		schema = jsonref.load(file)

	with open("src/evaluation/spec_comparison_pairs.json", "r", encoding="utf-8") as file:
		pairs = json.load(file)

	for pair in pairs:
		scores = calculate_eval_metrics(json.dumps(pair["spec1"]), json.dumps(pair["spec2"]), schema)
		print(pair["description_of_difference"])
		print(json.dumps(asdict(scores), indent=4))
		print()

if __name__ == "__main__":
	main()
