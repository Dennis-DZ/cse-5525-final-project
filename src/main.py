from argparse import ArgumentParser, Namespace
from datetime import datetime
from src.data_loader import (
	load_examples_from_file,
	load_inference_outputs_from_file
)
from src.utils import safe_open
from pathlib import Path
import src.evaluation.eval_outputs
import src.icl.inference as icl_inference
import logging

def get_args():
	global_parser = ArgumentParser()
	sub_parsers = global_parser.add_subparsers(dest="command", required=True)

	date_string = datetime.now().strftime("%Y-%m-%dT%H-%M")

	icl_inference_parser = sub_parsers.add_parser("icl_inference")
	add_icl_inference_args(icl_inference_parser, date_string)

	evaluate_parser = sub_parsers.add_parser("evaluate")
	add_evaluate_args(evaluate_parser)

	args = global_parser.parse_args()

	match args.command:
		case "icl_inference":
			if args.output_path is None:
				args.output_path = f"output/icl_inference/data/{args.experiment_name}.jsonl"
			if args.log_path is None:
				args.log_path = f"output/icl_inference/log/{args.experiment_name}.log"
		case "evaluate":
			input_filename = Path(args.input_path).stem
			if args.output_path is None:
				args.output_path = f"output/evaluation/data/{input_filename}.jsonl"
			if args.log_path is None:
				args.log_path = f"output/evaluation/log/{input_filename}.log"

	return args

def add_icl_inference_args(icl_inference_parser: ArgumentParser, date: str):
	icl_inference_parser.add_argument("--experiment_name", type=str, default=date)
	icl_inference_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
	icl_inference_parser.add_argument("--test_data_path", type=str, default="data/dev.jsonl")
	icl_inference_parser.add_argument("--num_test_examples", type=int, default=392)
	icl_inference_parser.add_argument("--system_prompt_path", type=str, default="src/icl/sample_system_prompt.txt")
	icl_inference_parser.add_argument("--user_prompt_path", type=str, default="src/icl/sample_user_prompt.txt")
	icl_inference_parser.add_argument("--output_path", type=str)
	icl_inference_parser.add_argument("--log_path", type=str)
	icl_inference_parser.add_argument("--seed", type=int, default=42)

def add_evaluate_args(evaluate_parser: ArgumentParser):
	evaluate_parser.add_argument("--input_path", type=str, required=True)
	evaluate_parser.add_argument("--output_path", type=str)
	evaluate_parser.add_argument("--log_path", type=str)
	evaluate_parser.add_argument("--schema_path", type=str, default="data/simulation_schema.json")

def run_icl_inference(args: Namespace):
	from numpy.random import seed as numpy_seed
	from random import seed as random_seed
	from torch import manual_seed as torch_seed

	random_seed(args.seed)
	numpy_seed(args.seed)
	torch_seed(args.seed)

	test_data = load_examples_from_file(args.test_data_path, args.num_test_examples)
	icl_inference.run_inference(
		model_name=args.model,
		system_prompt_path=args.system_prompt_path,
		user_prompt_path=args.user_prompt_path,
		output_path=args.output_path,
		test_data=test_data,
	)

def run_evaluation(args: Namespace):
	inference_outputs = load_inference_outputs_from_file(args.input_path)
	src.evaluation.eval_outputs.run_evaluation(
		inference_outputs,
		args.output_path,
		args.schema_path,
	)

def main():
	args = get_args()

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	with safe_open(args.log_path, "x"):
		file_handler = logging.FileHandler(args.log_path)
	formatter = logging.Formatter("{levelname} - {message}", style="{")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	logger.info(args)

	match args.command:
		case "icl_inference":
			run_icl_inference(args)
		case "evaluate":
			run_evaluation(args)

if __name__ == "__main__":
	main()
