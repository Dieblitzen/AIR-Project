import sys
sys.path.append('.')
from ImSeg_Dataset import ImSeg_Dataset
from ImSeg.train import model_from_config, calculate_iou_prec_recall

import os
import json
import logging
import argparse
import numpy as np
import tensorflow as tf


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to get validation/test set results"+
                                                 "on a trained Image Segmentation model.")
  parser.add_argument('--data_path',
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--set_type',
                      type=str,
                      default='val',
                      help='Run inference on either train/val/test images and annotations.')
  parser.add_argument('--config',
                      type=str,
                      required=True,
                      help='Path to model config .json file defining model hyperparams.')
  parser.add_argument('--checkpoint',
                      type=str,
                      default=None,
                      help='(Optional) path to checkpoint dir. If not given, will find based'+
                            ' on model name from config and given data_path.')
  parser.add_argument('--classes_path',\
                      type=str,
                      default=os.path.join('.', 'classes.json'),
                      help='Path to directory where defined classes are stored.')
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()
  set_type = args.set_type
  assert set_type in {"train", "val", "test"}, "Must specify one of train/val/test sets."
  checkpoint_path = args.checkpoint

  # Get args from config.
  config_path = args.config
  with open(config_path, 'r') as f:
    config = json.load(f)
  model_type = config.get("type", "RefineNet")
  model_name = config["name"]
  batch_size = config["batch_size"]

  ## Set up dataset
  dataset = ImSeg_Dataset(data_path=args.data_path, classes_path=args.classes_path)
  assert dataset.data_sizes[0] != 0, "Dataset should be built before inference."
  
  # Number of samples, number of batches and interested classes.
  sizes_index = {"train": 0, "val": 1, "test": 2}
  num_samples = dataset.data_sizes[sizes_index[set_type]]
  num_batches = num_samples//batch_size
  config["classes"] = dataset.seg_classes if not config["classes"] else config["classes"]
  interest_classes = config["classes"]

  # Create model output dir where preds will be stored. Save config here.
  dataset.create_model_out_dir(model_name)
  with open(os.path.join(dataset.model_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

  ## Load model from config, load weights
  checkpoint_path = checkpoint_path if checkpoint_path else dataset.checkpoint_path
  model = model_from_config(model_type, config)
  model.load_weights(os.path.join(checkpoint_path, model_name))

  ## Iterate over dataset.
  data_indices = list(range(num_samples))
  for batch in range(num_batches):
    iter_indices = data_indices[batch*batch_size : (batch+1)*batch_size]
    img_input, label_masks = dataset.get_batch(iter_indices, set_type, 
                                               classes_of_interset=interest_classes)

    # Feed inputs to model
    img_input = np.array(img_input, dtype=np.float32)
    preds = model(img_input)

    # Get metrics for each image in batch
    batch_metrics = []
    for i, (pred, label_mask) in enumerate(zip(preds.numpy(), label_masks)):
      pred = pred[np.newaxis, :]
      label_mask = label_mask[np.newaxis, :]

      iou, prec, recall = calculate_iou_prec_recall(pred, label_mask)

      metrics = {}
      for i, class_name in enumerate(interest_classes):
        metrics[f'class_{class_name}_iou'] = iou[i]
        metrics[f'class_{class_name}_prec'] = prec[i]
        metrics[f'class_{class_name}_recall'] = recall[i]

      batch_metrics.append(metrics)

    # Make pixel values between 0 and 1
    batch_preds = (preds.numpy() >= 0).astype(np.uint8)

    # Save preds
    dataset.save_preds(iter_indices, batch_preds, batch_metrics, set_type=set_type)
