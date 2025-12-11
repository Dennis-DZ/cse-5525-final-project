from dataclasses import dataclass
import json
import random

@dataclass
class DataExample:
	prompt: str
	spec: str

	@staticmethod
	def from_json(json_str: str):
		d = json.loads(json_str)
		return DataExample(d["prompt"], d["completion"])

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
