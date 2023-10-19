from argparse import ArgumentParser
import jsonlines
from collections import defaultdict as ddict
import numpy as np
import os
import json


def cal_acc(preds,labels):
    results = []
    for i in range(len(preds)):
        if preds[i] == labels[i] and preds[i] != -1:
            results.append(1)
        else:
            results.append(0)
    return round(np.sum(results)/len(results),2)


def main(args):

	d = ddict( lambda : ddict (list))
	all_labels = []
	all_preds = []
	with jsonlines.open(args.data_path) as f:
		for line in f:
			if line["source"].startswith("hagrid"):
				line["source"] = "hagrid"
			d[line["source"]]["labels"].append(line["postprocess_label"])
			d[line["source"]]["preds"].append(line["postprocess_output"])
			all_labels.append(line["postprocess_label"])
			all_preds.append(line["postprocess_output"])

	for key in d:
		d[key]["acc"] = cal_acc(d[key]["labels"],d[key]["preds"])
	d["all"]["acc"] = cal_acc(all_labels,all_preds)

	file_name, file_extension = os.path.splitext(args.data_path)
	data_path = f"{file_name}_analysis{file_extension}"
	with open(data_path,"w") as f:
		for key in d:
			json.dump({"source":key,"acc":d[key]["acc"]},f)
			f.write("\n")
	
if __name__ == "__main__":
	parser = ArgumentParser("analysis")
	parser.add_argument("--data_path",default="",type=str)
	args = parser.parse_args()
	main(args)