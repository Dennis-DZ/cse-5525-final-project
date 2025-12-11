from transformers import PreTrainedTokenizerFast, PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

def prompt_model(
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizerFast,
		system_prompt: str,
		user_prompt: str,
	) -> str:

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

	generated_ids = model.generate(
		**model_inputs,
		max_new_tokens=512
	)

	generated_ids = [
		output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
	]

	response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

	return response

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

	print(prompt_model(model, tokenizer, system_prompt, user_prompt))

if __name__ == "__main__":
	main()
