from collections import defaultdict as ddict
import json

d = ddict(dict)

d["base_llama"]["INSTRUCTION"] = ("### Instruction:\n"
        						"Please verify whether the reference can support the claim to the question. Options: 'attributable' or 'not attributable'.")
d["base_llama"]["INSTRUCTION"] += "\n"
d["base_llama"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["base_llama"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}"


d["base_longlora"]["INSTRUCTION"] = ("Below is an instruction that describes a task. "
									"Write a response that appropriately completes the request.\n\n"
									"### Instruction:\n"
									"Please verify whether the reference can support the claim to the question. Options: 'attributable' or 'not attributable'.")
d["base_longlora"]["INSTRUCTION"] += "\n"
d["base_longlora"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["base_longlora"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nReference: {}"






d["w_informativeness_w_response_llama"]["INSTRUCTION"] = ("### Instruction:\n"
        						"Your task is to evaluate if the given reference supports the claim in a response to a specific question. The whole response will be provided as a context, but you should focus on the single claim, which may be a subset of or the same as the response. Your judgement should be based on the criteria of 'attributability', which refers to whether the claim is supported by the provided references. Please note, you are not evaluating the 'informativeness' of the claim, which refers to whether the claim provides a clear and direct answer to the question. Your focus should solely be on 'attributability'. Your response should be either 'attributable' if it the claim is supported by the reference, or 'not attributable' if the reference contradicts the claim or there is not enough information to support the claim.")
d["w_informativeness_w_response_llama"]["INSTRUCTION"] += "\n"
d["w_informativeness_w_response_llama"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["w_informativeness_w_response_llama"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}"


d["w_informativeness_w_response_longlora"]["INSTRUCTION"] = ("Below is an instruction that describes a task. "
									"Write a response that appropriately completes the request.\n\n"
									"### Instruction:\n"
									"Your task is to evaluate if the given reference supports the claim in a response to a specific question. The whole response will be provided as a context, but you should focus on the single claim, which may be a subset of or the same as the response. Your judgement should be based on the criteria of 'attributability', which refers to whether the claim is supported by the provided references. Please note, you are not evaluating the 'informativeness' of the claim, which refers to whether the claim provides a clear and direct answer to the question. Your focus should solely be on 'attributability'. Your response should be either 'attributable' if it the claim is supported by the reference, or 'not attributable' if the reference contradicts the claim or there is not enough information to support the claim.")
d["w_informativeness_w_response_longlora"]["INSTRUCTION"] += "\n"
d["w_informativeness_w_response_longlora"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["w_informativeness_w_response_longlora"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}"






d["w_informativeness_w_response_w_longref_llama"]["INSTRUCTION"] = ("### Instruction:\n"
        						"Your task is to evaluate if the given reference supports the claim in a response to a specific question. The whole response will be provided as a context, but you should focus on the single claim, which may be a subset of or the same as the response. Your judgement should be based on the criteria of 'attributability', which refers to whether the claim is supported by the provided references. Please note, you are not evaluating the 'informativeness' of the claim, which refers to whether the claim provides a clear and direct answer to the question. Your focus should solely be on 'attributability'. Your response should be either 'attributable' if it the claim is supported by the reference, or 'not attributable' if the reference contradicts the claim or there is not enough information to support the claim.")
d["w_informativeness_w_response_w_longref_llama"]["INSTRUCTION"] += "\n"
d["w_informativeness_w_response_w_longref_llama"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["w_informativeness_w_response_w_longref_llama"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}"


d["w_informativeness_w_response_w_longref_longlora"]["INSTRUCTION"] = ("Below is an instruction that describes a task. "
									"Write a response that appropriately completes the request.\n\n"
									"### Instruction:\n"
									"Your task is to evaluate if the given reference supports the claim in a response to a specific question. The whole response will be provided as a context, but you should focus on the single claim, which may be a subset of or the same as the response. Your judgement should be based on the criteria of 'attributability', which refers to whether the claim is supported by the provided references. Please note, you are not evaluating the 'informativeness' of the claim, which refers to whether the claim provides a clear and direct answer to the question. Your focus should solely be on 'attributability'. Your response should be either 'attributable' if it the claim is supported by the reference, or 'not attributable' if the reference contradicts the claim or there is not enough information to support the claim.")
d["w_informativeness_w_response_w_longref_longlora"]["INSTRUCTION"] += "\n"
d["w_informativeness_w_response_w_longref_longlora"]["PROMPT_TEMPLATE"] = "{instruction}\n{input}\n### Response:\n"
d["w_informativeness_w_response_w_longref_longlora"]["INPUT_TEMPLATE"] = "### Input:\nQuestion: {}\n\nClaim: {}\n\nResponse: {}\n\nReference: {}"




with open("src/template.json","w") as f:
	json.dump(d,f,indent = 4)






