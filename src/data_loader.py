from dataclasses import dataclass, field
from typing import Self
import json
import random
import math

@dataclass
class DataExample:
	prompt: str
	spec: str

	@staticmethod
	def from_json(json_str: str):
		d = json.loads(json_str)
		return DataExample(d["prompt"], d["completion"])
	
@dataclass
class InferenceOutput:
	nl_prompt: str = ""
	true_spec: str = ""
	pred_spec: str = ""
	num_tokens: int = 0
	inference_time: float = 0

	@staticmethod
	def from_json(json_str: str):
		return InferenceOutput(**json.loads(json_str))
	
@dataclass
class EvaluationOutput:
	inference_output: InferenceOutput = field(default_factory=InferenceOutput)
	valid_json: bool = False
	schema_match: bool = False
	schema_error_message: str = ""
	schema_error_path: str = ""
	key_f1: float = 0
	key_precision: float = 0
	key_recall: float = 0
	item_f1: float = 0
	item_precision: float = 0
	item_recall: float = 0
	tree_edit_similarity: float = 0
	tree_edit_distance: float = math.inf

	def add(self, a: Self) -> None:
		self.inference_output.inference_time += a.inference_output.inference_time
		self.key_f1 += a.key_f1
		self.key_precision += a.key_precision
		self.key_recall += a.key_recall
		self.item_f1 += a.item_f1
		self.item_precision += a.item_precision
		self.item_recall += a.item_recall
		self.tree_edit_similarity += a.tree_edit_similarity

def load_examples_from_file(
		data_examples_path: str,
		num_to_load: int,
) -> list[DataExample]:

	with open(data_examples_path, "r") as file:
		data_lines = file.readlines()

	sampled_data_lines = random.sample(data_lines, num_to_load)
	loaded_data_examples = [
		DataExample.from_json(line) for line in sampled_data_lines
	]

	return loaded_data_examples

def load_inference_outputs_from_file(inference_outputs_path: str):
	with open(inference_outputs_path, "r") as file:
		loaded_inference_outputs = [
			InferenceOutput.from_json(line) for line in file.readlines()
		]

	return loaded_inference_outputs
