import sys
sys.path.append('.')
import json
import ImSeg.refine_net as refine_net

import os
import numpy as np
import tensorflow as tf

## Supported model variants, along with model loading function given a config dictionary.
MODEL_TYPES = {"RefineNet": refine_net.refine_net_from_config}


"""
Initialises image segmentation model given a config dictionary.
Will load in checkpoint weights if specified.
Requires: 
  config: A valid config dictionary for the type of model
"""
def load_model(config, from_checkpoint=None):
  model_type = config.get("type", "RefineNet")
  assert model_type in MODEL_TYPES, "Input model type is not supported yet."
  model = MODEL_TYPES[model_type](config)
  
  if from_checkpoint:
    model_name = config.get("name")
    checkpoint_path = os.path.join(from_checkpoint, model_name)
    model.load_weights(checkpoint_path)

  return model


"""
Saves an image segmentation model given a model config and
the checkpoint directory.
Requires:
  `model`: The model to be saved.
  `config`: A valid config dictionary for the type of model
  `checkpoint_dir`: Path to directory where weights will be saved.
"""
def save_model(model, config, checkpoint_dir):
  model_name = config["name"]
  model.save_weights(os.path.join(checkpoint_dir, model_name))
