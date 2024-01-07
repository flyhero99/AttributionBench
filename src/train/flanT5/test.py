from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import torch

device = "cuda"
tokenizer = T5Tokenizer.from_pretrained("./t5-large-combined/")
model = T5ForConditionalGeneration.from_pretrained("./t5-large-combined/").to(device)

def eval(test_data, model_responses):
	att_tp = 0
	con_tp = 0
	exp_tp = 0

	att_fp = 0
	con_fp = 0
	exp_fp = 0
	
	att_fn = 0
	con_fn = 0
	exp_fn = 0
	for gold, pred in zip(test_data, model_responses):
		if gold["output"] == "Attributable":
			if pred["response"].startswith(gold["output"]):
				att_tp += 1
			else:
				att_fn += 1
				if pred["response"].startswith("Contradictory"):
					con_fp += 1
				elif pred["response"].startswith("Extrapolatory"):
					exp_fp += 1
		elif gold["output"] == "Contradictory":
			if pred["response"].startswith(gold["output"]):
				con_tp += 1
			else:
				con_fn += 1
				if pred["response"].startswith("Attributable"):
					att_fp += 1
				elif pred["response"].startswith("Extrapolatory"):
					exp_fp += 1
		elif gold["output"] == "Extrapolatory":
			if pred["response"].startswith(gold["output"]):
				exp_tp += 1
			else:
				exp_fn += 1
				if pred["response"].startswith("Attributable"):
					att_fp += 1
				elif pred["response"].startswith("Contradictory"):
					con_fp += 1
	
	att_p = att_tp / (att_tp + att_fp)
	att_r = att_tp / (att_tp + att_fn)
	att_f = (2 * att_p * att_r) / (att_p + att_r + 1e-8)
	print(
		"Attributable: P {:.4f} R {:.4f} F1 {:.4f}".format(
			att_p, att_r, att_f
		)
	)
	
	con_p = con_tp / (con_tp + con_fp)
	con_r = con_tp / (con_tp + con_fn)
	con_f = (2 * con_p * con_r) / (con_p + con_r + 1e-8)
	print(
		"Contradictory: P {:.4f} R {:.4f} F1 {:.4f}".format(
			con_p, con_r, con_f
		)
	)
	
	exp_p = exp_tp / (exp_tp + exp_fp)
	exp_r = exp_tp / (exp_tp + exp_fn)
	exp_f = (2 * exp_p * exp_r) / (exp_p + exp_r + 1e-8)
	print(
		"Extrapolatory: P {:.4f} R {:.4f} F1 {:.4f}".format(
			exp_p, exp_r, exp_f
		)
	)

	print("Overall Accuracy: {:.4f}".format((att_tp + con_tp + exp_tp) / len(test_data)))
	

def inference(test_data, fname):
	results = []
	for example in tqdm(test_data):
		prompt = example['instruction'] + "\n\n" + example['input']
		batch = tokenizer(prompt, return_tensors="pt", truncation=True)
		batch = {k: v.to(device) for k, v in batch.items()}

		with torch.no_grad():
			completion = model.generate(
				inputs=batch["input_ids"],
				attention_mask=batch["attention_mask"],
				# temperature=0.7,
				# top_p=0.9,
				# do_sample=True,
				num_beams=1,
				max_new_tokens=8,
				eos_token_id=tokenizer.eos_token_id,
				pad_token_id=tokenizer.pad_token_id
			)

		response = tokenizer.decode(completion[0], skip_special_tokens=True)
		results.append({"prompt": prompt, "response": response})

	out = open(fname, "w+")
	json.dump(results, out, indent=2)
	out.close()

in_fname = "./attreval-gensearch.json"
out_fname = "t5-large-combined-gensearch.json"

test_data = json.load(open(in_fname))
inference(test_data, out_fname)
model_responses = json.load(open(out_fname))
eval(test_data, model_responses)