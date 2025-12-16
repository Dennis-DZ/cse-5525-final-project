from transformers import (
	PreTrainedTokenizerFast,
	PreTrainedModel,
	AutoModelForCausalLM,
	AutoTokenizer,
	GenerationMixin
)
from src.data_loader import DataExample
from src.utils import safe_open
from typing import cast
import torch.cuda
import logging
import json

def prompt_model(
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerFast,
		system_prompt: str,
		user_prompt: str,
) -> tuple[str, int, float]:

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt}
	]

	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True
	)

	model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

	start_event = torch.cuda.Event(enable_timing=True)
	end_event = torch.cuda.Event(enable_timing=True)

	start_event.record()
	generated_ids = cast(GenerationMixin, model).generate(
		**model_inputs,
		max_new_tokens=1024
	)
	end_event.record()

	torch.cuda.synchronize()
	time_ms = start_event.elapsed_time(end_event)

	new_generated_ids = [
		output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
	]
	num_tokens = len(new_generated_ids[0])

	response = tokenizer.batch_decode(new_generated_ids, skip_special_tokens=True)[0]

	return response, num_tokens, time_ms

def run_inference(
		model_name: str,
		system_prompt_path: str,
		user_prompt_path: str,
		output_path: str,
		test_data: list[DataExample],
) -> None:
	logger = logging.getLogger("__main__")
	logger.info("Starting ICL inference")

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype="auto",
		device_map="auto"
	)

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	with open(system_prompt_path, "r") as file:
		system_prompt = file.read()
	logger.info(f"System prompt:\n```\n{system_prompt}```\n")

	with open(user_prompt_path, "r") as file:
		user_prompt_template = file.read()
	logger.info(f"User prompt template:\n```\n{user_prompt_template}```\n")

	num_examples = len(test_data)

	with safe_open(output_path, "x"):
		# Create necessary directories and ensure output_path is a new file
		pass

	for i, test_data_example in enumerate(test_data):
		logger.info(f"{i+1}/{num_examples}")

		user_prompt = user_prompt_template.replace("%new_prompt%", test_data_example.prompt)
		response, num_tokens, time = prompt_model(model, tokenizer, system_prompt, user_prompt)

		output = dict(
			nl_prompt=test_data_example.prompt,
			true_spec=test_data_example.spec,
			pred_spec=response,
			num_tokens=num_tokens,
			inference_time=time,
		)

		with open(output_path, "a") as file:
			json.dump(output, file)
			file.write("\n")

	logger.info("Finished ICL inference")

def main():
	from src.data_loader import load_examples_from_file
	from datetime import datetime
	import random

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	console_handler = logging.StreamHandler()
	formatter = logging.Formatter("{levelname} - {message}", style="{")
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)

	random.seed(42)
	test_data = load_examples_from_file("data/dev.jsonl", num_to_load=5)

	date_string = datetime.now().strftime("%Y-%m-%dT%H-%M")

	run_inference(
		model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
		system_prompt_path="src/icl/sample_system_prompt.txt",
		user_prompt_path="src/icl/sample_user_prompt.txt",
		output_path=f"output/icl_inference/data/test-{date_string}.jsonl",
		test_data=test_data,
	)

if __name__ == "__main__":
	main()
