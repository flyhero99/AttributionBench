import json
import pandas as pd
import os
import json
import utils
from tqdm import tqdm
import re
import uuid


class DatasetLoader():
    def __init__(self):
        # self.dataset_name = dataset_name
        self.dataset_list = [
            "ASQA", "AttributedQA", "HAGRID", "QAMPARI", "StrategyQA", "SUMMEDITS",
            "ExpertQA", "AttrScore-Gensearch", "Stanford-Gensearch",
        ]

    def check_empty_references(self, references):
        for x in references:
            if x in ["", None]:
                return True
        return False
    
    def load_dataset(self, dataset_name, split="train"):
        try:
            if dataset_name.lower() in [x.lower() for x in self.dataset_list]:
                pass
        except:
            print(f"Dataset {dataset_name} not supported or not downloaded yet.")

        data = []
        print(f"Loading data from {dataset_name}.")
        
        # ASQA train set (to be implemented)
        if dataset_name.lower() == "ASQA".lower():
            data = []  # to be implemented

        # AttributedQA train set
        elif dataset_name.lower() == "AttributedQA".lower():
            df = pd.read_csv("../our_data/AttributedQA/{}.csv".format(split))
            selected = df.groupby('question').apply(lambda x: x.head(300)).reset_index(drop=True)
            for i in range(selected.shape[0]):
                question = selected.iloc[i]["question"]
                claim = selected.iloc[i]["answer"]
                documents = [selected.iloc[i]["passage"]]
                data_item_to_add = {
                    "question": question,
                    "claim": str(claim),
                    "claim_raw_string": str(claim),
                    "response": str(claim),
                    "references": documents,
                    "citation_links": [],
                    "webpage_references": [],
                    "attribution_label": "attributable" if selected.iloc[i]["human_rating"] == "Y" else "not attributable",
                    "src_dataset": "AttributedQA",
                }
                data_item_to_add["id"] = "AttributedQA_" + str(uuid.uuid4())
                data.append(data_item_to_add)

        # HAGRID train set and dev set
        elif dataset_name.lower() == "HAGRID".lower():
            df = [json.loads(line) for line in open("../our_data/hagrid/{}.jsonl".format(split))]
            
            for data_item in df:
                question = data_item["query"]
                documents_list = [_["text"] for _ in data_item["quotes"]]
                for answer in data_item["answers"]:
                    if "attributable" in answer:  # only keep instances with attributable labels
                        response = answer["answer"]
                        for sentence in answer["sentences"]:
                            claim = sentence["text"]
                            pattern = r'\[(\d+(?:,\s*\d+)*)\]'
                            matches = re.findall(pattern, claim)  # remove citations like [1][2] or [1, 2]
                            doc_indices = [int(num) for match in matches for num in match.split(",")]
                            doc_indices = list(set(doc_indices))
                            documents = ["[{}] {}".format(doc_index, documents_list[doc_index-1]) for doc_index in doc_indices \
                                        if len(doc_indices) > 0 and doc_index-1 < len(documents_list)]
                            
                            if "attributable" in sentence:  # only care about those data with attribution labels
                                data_item_to_add = {
                                    "question": question,
                                    "claim": claim,
                                    "claim_raw_string": claim,
                                    "response": response,
                                    "references": documents,
                                    "citation_links": [],
                                    "webpage_references": [],
                                    "attribution_label": "attributable" if str(sentence["attributable"]) == "1" else "not attributable",
                                    "src_dataset": "HAGRID",
                                }
                                data_item_to_add["id"] = "HAGRID_" + str(uuid.uuid4())
                                data.append(data_item_to_add)
                
        elif dataset_name.lower() == "ExpertQA".lower():
            df = [json.loads(line) for line in open("../our_data/ExpertQA/{}.jsonl".format(split))]

            for idx, data_item in enumerate(df):
                question = data_item["question"]  # 提取 question
                answers = data_item["answers"]  # 提取 answers
                k = list(data_item['answers'].keys())[0]  # 数据的第一个 key 代表提供回答的模型名，因此需要把这个 key 先取出来
                answers_body = answers[k]  # 实际的 answer body
                response = answers_body["answer_string"]  # 提取 response
                claims = answers_body["claims"]  # answer 中的所有 claims
                # attribution_webpage_contents = answers_body["attribution_webpage_content"]
                attribution_webpage_contents = []
                for claim in claims:
                    claim_string = claim.get('claim_string', None)
                    evidences = claim.get('evidence', None)
                    documents = []  # 数据中原有的 evidence 字段
                    citation_links = []  # 本条 claim 引用过的链接
                    webpage_documents = []  # 本条 claim 引用过的链接对应的网页内容

                    for evidence in evidences:
                        citation_number_match = re.search(r'\[(\d+)\]', evidence)
                        citation_text_match = re.search(r'\[\d+\]', evidence)
                        citation_number = int(citation_number_match.group(1))
                        citation_text = citation_text_match.group()

                        # 添加 evidence 作为 references，格式：[1] This is an apple ...
                        if evidence:
                            documents.append(evidence)
                        else:
                            documents.append("")

                        # 添加 citation urls
                        if answers_body["attribution"][citation_number-1]:
                            citation_links.append(answers_body["attribution"][citation_number-1].split(" ")[-1])
                        else:
                            citation_links.append("")
                            
                        # # 添加 webpage 作为 references，格式：[1] This is an apple ...
                        # if attribution_webpage_contents[citation_number-1] not in ["", None]:
                        #     webpage_documents.append("{} {}".format(citation_text, \
                        #                                     attribution_webpage_contents[citation_number-1]))
                        # else:
                        #     webpage_documents.append("")

                        data_item_to_add = {
                            "question": question,
                            "claim": claim_string,
                            "claim_raw_string": claim_string,
                            "response": response,
                            "references": documents,
                            "citation_links": citation_links,
                            "webpage_references": [],
                            "attribution_label": "attributable" if str(claim["support"]) == "Complete" else "not attributable",
                            "src_dataset": "ExpertQA",
                        }
                        data_item_to_add["id"] = "ExpertQA_" + str(uuid.uuid4())
                        if not self.check_empty_references(data_item_to_add["references"]):
                            data.append(data_item_to_add)

        elif dataset_name.lower() == "AttrScore-Gensearch".lower():
            df = pd.read_csv("../our_data/AttrScore-GenSearch/AttrEval-GenSearch.csv")
            for i in range(df.shape[0]):
                question = df.iloc[i]["query"]
                answer = str(df.iloc[i]["answer"])
                documents = [str(df.iloc[i]["reference"])]

                data_item_to_add = {
                    "question": question,
                    "claim": answer,
                    "claim_raw_string": answer,
                    "response": answer,
                    "references": documents,
                    "citation_links": [],
                    "webpage_references": [],
                    "attribution_label": "attributable" if df.iloc[i]["label"] == "Attributable" else "not attributable",
                    "src_dataset": "AttrScore-GenSearch",
                }
                data_item_to_add["id"] = "AttrScore-GenSearch_" + str(uuid.uuid4())
                data.append(data_item_to_add)

        elif dataset_name.lower() == "Stanford-GenSearch".lower():
            df = [json.loads(line) for line in open("../our_data/Stanford-GenSearch/{}.jsonl".format(split))]
            for data_item in tqdm(df):
                # 提取'query'和'response'字段
                question = data_item["query"]
                response = data_item['response']
                # 提取'statement_to_annotation'字段
                statements = data_item['annotation']['statement_to_annotation']
                # 提取'statements_to_citation_texts'字段
                statements_to_citation_texts = data_item['statements_to_citation_texts']

                citations_text_to_citation = {}
                for citation in data_item['citations']:
                    if citation['text'] in citations_text_to_citation:
                        continue
                    else:
                        citations_text_to_citation[citation['text']] = citation

                # 遍历'statement_to_annotation'的每一个键值对
                for statement, annotation in statements.items():
                    # 检查'statement_supported'字段，不为 None 的话保留这条数据
                    if annotation['statement_supported'] is not None:
                        documents = []  # 数据中原有的 evidence 字段
                        citation_links = []  # 本条 claim 引用过的链接
                        webpage_documents = []  # 本条 claim 引用过的链接对应的网页内容

                        # 如果存在'citation_annotations'，则提取'citation_supports'和'evidence'字段
                        if annotation['citation_annotations']:
                            for citation in annotation['citation_annotations']:
                                citation_text = citation["citation_text"]  # 格式: [1], [2], ...

                                # 添加 evidence 作为 references，格式：[1] This is an apple ...
                                if citation['evidence']:
                                    documents.append("{} {}".format(citation_text, citation['evidence']))
                                else:
                                    documents.append("")

                                # 添加 citation urls
                                if citations_text_to_citation[citation_text]["link_target"]:
                                    citation_links.append(citations_text_to_citation[citation_text]["link_target"])
                                else:
                                    citation_links.append("")
                                    
                                # # 添加 webpage 作为 references，格式：[1] This is an apple ...
                                # if citations_text_to_citation[citation_text]["link_target_webpage_content"]:
                                #     webpage_documents.append("{} {}".format(citation_text, \
                                #                             citations_text_to_citation[citation_text]["link_target_webpage_content"]))
                                # else:
                                #     webpage_documents.append("")
                        else:
                            print("error!")
                            break
                        # pattern = r'\[(\d+(?:,\s*\d+)*)\]'
                        # matches = re.findall(pattern, statement)  # remove citations like [1][2] or [1, 2]
                        # cleaned_statement = re.sub(pattern, '', statement)
                        data_item_to_add = {
                            "question": question,
                            "claim": statement,
                            "claim_raw_string": statement,
                            "response": response,
                            "references": documents,
                            "citation_links": citation_links,
                            "webpage_references": webpage_documents,
                            "attribution_label": "attributable" if annotation['statement_supported'] and annotation['statement_supported'] == "Yes" else "not attributable",
                            "src_dataset": "Stanford-GenSearch",
                        }
                        data_item_to_add["id"] = "Stanford-GenSearch_" + str(uuid.uuid4())
                        if True or not self.check_empty_references(data_item_to_add["references"]):
                            data.append(data_item_to_add)
          
        else:
            print(f"Dataset {dataset_name} not supported or not downloaded yet.")

        seen = set()
        unique_data = []

        for item in data:
            key = item["question"] + item["claim"] + item["response"] + "".join(item["references"])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        print(f"Dataset Name: {dataset_name}, total items: {len(data)}, total unique <question, claim, references> pairs: {len(unique_data)}")
        return unique_data
        # return data

def dedup_json_list(arr, field_name="query"):
    s = set()
    res = []
    for x in arr:
        if x[field_name] not in s:
            s.add(x[field_name])
            res.append(x)
    return res