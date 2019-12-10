import sys
sys.path.append('.')
import json
from ImSeg_Dataset import ImSeg_Dataset
import ImSeg.refine_net as refine_net
import ImSeg.resnet as resnet

import os
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
  parser.add_argument('--config',
                      type=str,
                      required=True,
                      help='Path to model config .json file defining model hyperparams.')
  parser.add_argument('--classes_path',\
                      type=str,
                      default=os.path.join('.', 'classes.json'),
                      help='Path to directory where defined classes are stored.')
  args = parser.parse_args()
  return args