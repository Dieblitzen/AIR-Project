import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def passed_arguments():
	parser = argparse.ArgumentParser(description="Script to evaluate model predictions.")
	parser.add_argument("--preds",
											type=str,
											required=True,
											help="Path to dir containing model pred images and metrics.")
	args = parser.parse_args()
	return args


def evaluate(path_to_pred_dir):
	"""
	Prints the mean metrics for each evaluated prediction in the pred_dir.
	"""
	metric_files = os.listdir(path_to_pred_dir)
	metric_files = [f for f in metric_files if f.endswith('.json')]

	aggregate = {}
	n = 0
	for metric_path in metric_files:
		with open(os.path.join(path_to_pred_dir, metric_path), 'r') as f:
			metric_dict = json.load(f)
		for metric in metric_dict:
			aggregate[metric] = aggregate.get(metric, 0) + metric_dict[metric]
		n += 1
    
	print(path_to_pred_dir)
	for metric_name, metric in aggregate.items():
		print(f'Mean {metric_name}: {metric/n}')


if __name__ == "__main__":
	args = passed_arguments()
	path_to_pred_dir = args.preds

	evaluate(path_to_pred_dir)