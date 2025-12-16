from argparse import ArgumentParser, Namespace
from datetime import datetime
from src.data_loader import load_examples_from_file
from numpy.random import seed as numpy_seed
from random import seed as random_seed
from torch import manual_seed as torch_seed
from src.utils import safe_open
import src.icl.inference as icl_inference
import logging

def get_args():
	global_parser = ArgumentParser()
	sub_parsers = global_parser.add_subparsers(dest="command", required=True)

	date_string = datetime.now().strftime("%Y-%m-%dT%H-%M")

	icl_inference_parser = sub_parsers.add_parser("icl_inference")
	icl_inference_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
	icl_inference_parser.add_argument("--test_data_path", type=str, default="data/dev.jsonl")
	icl_inference_parser.add_argument("--num_test_examples", type=int, default=392)
	icl_inference_parser.add_argument("--system_prompt_path", type=str, default="src/icl/sample_system_prompt.txt")
	icl_inference_parser.add_argument("--user_prompt_path", type=str, default="src/icl/sample_user_prompt.txt")
	icl_inference_parser.add_argument("--output_path", type=str, default=f"output/icl_inference/data/{date_string}.jsonl")
	icl_inference_parser.add_argument("--log_path", type=str, default=f"output/icl_inference/log/{date_string}.log")
	icl_inference_parser.add_argument("--seed", type=int, default=42)

	return global_parser.parse_args()

def run_icl_inference(args: Namespace):
	test_data = load_examples_from_file(args.test_data_path, args.num_test_examples)
	icl_inference.run_inference(
		model_name=args.model,
		system_prompt_path=args.system_prompt_path,
		user_prompt_path=args.user_prompt_path,
		output_path=args.output_path,
		test_data=test_data,
	)

def main():
	args = get_args()

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	with safe_open(args.log_path, "a"):
		file_handler = logging.FileHandler(args.log_path)
	formatter = logging.Formatter("{levelname} - {message}", style="{")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	logger.info(args)

	random_seed(args.seed)
	numpy_seed(args.seed)
	torch_seed(args.seed)

	match args.command:
		case "icl_inference":
			run_icl_inference(args)

if __name__ == "__main__":
	main()
