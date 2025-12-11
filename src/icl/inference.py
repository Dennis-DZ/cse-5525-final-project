from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
import torch.cuda

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
	generated_ids = model.generate(
		**model_inputs,
		max_new_tokens=1024
	)
	end_event.record()

	torch.cuda.synchronize()
	time_ms = start_event.elapsed_time(end_event)

	generated_ids = [
		output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
	]
	num_tokens = len(generated_ids[0])

	response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

	return response, num_tokens, time_ms

def main():
	model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		dtype="auto",
		device_map="auto"
	)

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	with open("src/icl/sample_system_prompt.txt", "r") as file:
		system_prompt = file.read()

	with open("src/icl/sample_user_prompt.txt", "r") as file:
		user_prompt = file.read()

	response, num_tokens, time = prompt_model(model, tokenizer, system_prompt, user_prompt)

	print(response)
	print("Number of tokens:", num_tokens)
	print("Inference time (ms):", time)

if __name__ == "__main__":
	main()
