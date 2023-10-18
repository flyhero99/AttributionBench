import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import json
from collections import defaultdict as ddict
import pickle
import jsonlines
from tqdm import tqdm
import hashlib
from datasets import load_dataset
from retriever_normalize import normalize
from retriever_splitter import split_long_sentence, regex


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def unflatten(original, flat_scores):
    unflattened = []
    new_index = 0
    for index, item in enumerate(original):
        unflattened.append(flat_scores[new_index : new_index + len(item)])
        new_index += len(item)
    return unflattened

def get_batch(l,bs):
    for i in range(0,len(l),bs):
        yield l[i: i + bs]

# 注意token type _id
def tokenize_and_chunk(text, chunk_size ,tokenizer):
    # smaller than 512
    assert chunk_size <= 512 - 2
    tokens = tokenizer(text)["input_ids"]
    return [tokenizer.decode(tokens[i:i + chunk_size],skip_special_tokens= True) for i in range(0, len(tokens), chunk_size)]



def tokenize_and_truncate(text, tokenizer):
    # smaller than 512
    tokens = tokenizer(text,truncation = True,max_length = 512)["input_ids"]
    return tokenizer.decode(tokens,skip_special_tokens= True)



parser = argparse.ArgumentParser()

parser.add_argument("--method",choices=["dragon","contriever"],default="contriever")
parser.add_argument("--bs",type =int, default = 2)
parser.add_argument("--large_bs",type =int, default = 500)
parser.add_argument("--query_setting",nargs = "+", type = str)
parser.add_argument("--which_source",type = str, default="ExpertQA")
parser.add_argument("--output_dir",type =str, default = "")
parser.add_argument("--data_dir",type =str)
parser.add_argument("--dataset_version")
parser.add_argument("--chunk_size", type = int, default = 200)

args = parser.parse_args()
print(args.query_setting)
if args.output_dir == "":
    args.output_dir = f"{args.method}_{args.dataset_version}_{'|'.join(args.query_setting)}_{args.which_source}" 

from pathlib import Path
Path(args.output_dir).mkdir(exist_ok= True ,parents= True)
output_embed_path = os.path.join(args.output_dir,f"retrieval_embed_{args.method}.jsonl")
output_embed_clean_path = os.path.join(args.output_dir,f"retrieval_clean_{args.method}.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"


exist_line = 0
if os.path.exists(output_embed_path):
    with jsonlines.open(output_embed_path) as f:
        for line in f:
            exist_line += 1


# Example long contexts and queries
long_contexts = [
    "Your long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYourYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here ssdf Your long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here",
    "Your long context 2 here",
    "Your long context 3 here",
    "Your long context 4 here"
]

queries = [
    "Your long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYourYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here ssdf Your long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 hereYour long context 1 here",
    "Your query 2 here",
    "Your query 3 here",
    "Your query 4 here"
]



queries = []
long_contexts = []
ids = []
print("Before read")
data_dir = args.data_dir
datas = load_dataset(data_dir, split = "test", name = args.dataset_version)
datas = datas.filter(lambda example: example["src_dataset"] == args.which_source)
for example in datas:
    # question = example['question'] if example['question'] and example['question'] not in ["nan",""] else ""
    # claim = example['claim'] if example['claim'] and example['claim'] not in ["nan",""] else ""
    # response = example['response'] if example['response'] and example['response'] not in ["nan",""] else ""
    
    web_documents_concatenation = ".".join(example["webpage_references"])
    s = ""
    for name in args.query_setting:
        s += example[name]
    queries.append(s)
    ids.append(example['id'])
    long_contexts.append(web_documents_concatenation)


print("After read")
long_contexts_initial = long_contexts[exist_line:]
queries_initial = queries[exist_line:]
ids_initial = ids[exist_line:]
print(len(long_contexts_initial))
print(len(queries_initial))

large_bs = args.large_bs
bs = args.bs


pbar_total = tqdm(total = len(queries_initial),desc = "large")
for large_bs_index in range(0,len(queries_initial),large_bs):
    pbar_total.update(large_bs)

    queries_text = queries_initial[large_bs_index: large_bs_index + large_bs]
    long_contexts = long_contexts_initial[large_bs_index: large_bs_index + large_bs]
    ids = ids_initial[large_bs_index: large_bs_index + large_bs]
    chunked_contexts = [split_long_sentence(normalize(context),regex,chunk_size = args.chunk_size) for context in long_contexts]
    # chunked_contexts = [tokenize_and_chunk(context, 500, tokenizer) for context in long_contexts]
    flatten_chunked_contexts = flatten(chunked_contexts)
    flatten_chunked_contexts = [_.strip() for _ in flatten_chunked_contexts]

    all_query = []
    all_ctx = []
    pbar_quries = tqdm(total= len(queries_text),desc="for loop for queries")
    pbar_context = tqdm(total= len(flatten_chunked_contexts),desc="for loop for flatten_chunked_contexts")

    if args.method == "dragon":
        tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        query_encoder = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').to(device)
        context_encoder = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').to(device)
        truncated_queries = [tokenize_and_truncate(_,tokenizer) for _ in queries_text]
        
        for truncated_queries_batch in get_batch(truncated_queries,bs):
            query_input = tokenizer(truncated_queries_batch, padding=True, truncation=True, return_tensors='pt',max_length = 512).to(device)
            query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_query.append(query_emb)
            pbar_quries.update(bs)

        for context_batch in get_batch(flatten_chunked_contexts,bs):
            ctx_input = tokenizer(context_batch, padding=True, truncation=True, return_tensors='pt',max_length = 512).to(device)
            ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_ctx.append(ctx_emb)
            pbar_context.update(bs)

    if args.method == "contriever":

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever').to(device)
        truncated_queries = [tokenize_and_truncate(_,tokenizer) for _ in queries_text]

        for truncated_queries_batch in get_batch(truncated_queries,bs):
            query_input = tokenizer(truncated_queries_batch, padding=True, truncation=True, return_tensors='pt',max_length = 512).to(device)
            query_emb = model(**query_input)
            query_emb = mean_pooling(query_emb[0], query_input['attention_mask']).detach().cpu().numpy()
            all_query.append(query_emb)
            pbar_quries.update(bs)

        for context_batch in get_batch(flatten_chunked_contexts,bs):
            ctx_input = tokenizer(context_batch, padding=True, truncation=True, return_tensors='pt',max_length = 512).to(device)
            ctx_emb = model(**ctx_input)
            ctx_emb = mean_pooling(ctx_emb[0], ctx_input['attention_mask']).detach().cpu().numpy()
            all_ctx.append(ctx_emb)
            pbar_context.update(bs)

    if len(all_query) == 0:
        continue

    all_query = np.vstack(all_query).astype(float)
    all_ctx = np.vstack(all_ctx).astype(float)
    all_ctx_l = [all_ctx[i,:] for i in range(all_ctx.shape[0])]
    context_nested_embeddings = unflatten(chunked_contexts,all_ctx_l)
    all_embeddings = ddict( lambda : ddict(list))

    all_embeddings = ddict(list)
    for i in range(all_query.shape[0]):
        all_embeddings = ddict(list)
        scores = []
        try:
            for context_index in range(len(context_nested_embeddings[i])):
                scores.append(np.dot(context_nested_embeddings[i][context_index],all_query[i]))
        except:
            print(len(context_nested_embeddings[i]))
            print(all_query.shape)
            print(i)


        sorted_index = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True) 
        all_embeddings["score"] = scores
        all_embeddings["sorted_index"] = sorted_index
        all_embeddings["trun_query"] = [truncated_queries[i]]
        all_embeddings["context"] = [chunked_contexts[i]]
        all_embeddings["complete_query"] = [queries_text[i]]
        all_embeddings["id"] = ids[i]

        with open(output_embed_clean_path,"a") as f:
            json.dump(all_embeddings,f)
            f.write("\n")

        all_embeddings["trun_query_embed"] = [all_query[i,:].tolist()]
        context_nested_embeddings_l = [context_nested_embeddings[i][j].tolist() for j in range(len(context_nested_embeddings[i]))]
        all_embeddings["context_embed"] = context_nested_embeddings_l

        with open(output_embed_path,"a") as f:
            json.dump(all_embeddings,f)
            f.write("\n")

