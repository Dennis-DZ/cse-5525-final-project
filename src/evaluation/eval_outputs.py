from src.data_loader import InferenceOutput, EvaluationOutput
from src.utils import safe_open
from src.evaluation.evaluation import calculate_eval_metrics
from dataclasses import asdict
import logging
import jsonref # type: ignore
import json

def run_evaluation(
		inference_outputs: list[InferenceOutput],
		output_path: str,
		schema_path: str,
):
	logger = logging.getLogger("__main__")
	logger.info("Starting evaluation of inference outputs")

	with open(schema_path, "r") as file:
		schema = jsonref.load(file)

	with safe_open(output_path, "x"):
		# Create necessary directories and ensure output_path is a new file
		pass

	num_outputs = len(inference_outputs)
	num_valid_json = 0
	num_schema_match = 0
	sum = EvaluationOutput()

	for i, inference_output in enumerate(inference_outputs):
		logger.info(f"{i+1}/{num_outputs}")

		output = calculate_eval_metrics(
			inference_output.true_spec,
			inference_output.pred_spec,
			schema
		)

		output.inference_output = inference_output

		with open(output_path, "a") as file:
			json.dump(asdict(output), file)
			file.write("\n")

		sum.add(output)
		if output.valid_json:
			num_valid_json += 1
		if output.schema_match:
			num_schema_match += 1

	averages = dict(
		inference_time=sum.inference_output.inference_time / num_outputs,
		valid_json=num_valid_json / num_outputs,
		schema_match=num_schema_match / num_outputs,
		key=dict(
			f1=sum.key_f1 / num_outputs,
			precision=sum.key_precision / num_outputs,
			recall=sum.key_recall / num_outputs,
		),
		item=dict(
			f1=sum.item_f1 / num_outputs,
			precision=sum.item_precision / num_outputs,
			recall=sum.item_recall / num_outputs,
		),
		tree_edit_similarity=sum.tree_edit_similarity / num_outputs,
	)

	logger.info("averages: " + json.dumps(averages, indent=4))
	logger.info("Finished evaluation of inference outputs")
