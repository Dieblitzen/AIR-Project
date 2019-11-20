import sys
sys.path.append('.')
from ImSeg_Dataset import ImSeg_Dataset
import ImSeg.network as network
import ImSeg.backbone as backbone

import os
import argparse
import numpy as np
import tensorflow as tf


"""
Calculate IoU, Precision and Recall per class for entire batch of images.
Requires:
  preds: model preds array, shape (batch, h, w, #c)
  label_masks: ground truth masks, shape (batch, h, w, #c)
  pred_threshold: Confidence threshold over which pixel prediction counted,
"""
def calculate_iou_prec_recall(preds, label_masks, pred_threshold=0.5):
  # Reduce dimensions across all but classes dimension.
  preds = preds.reshape(-1, preds.shape[-1])
  label_masks = label_masks.reshape(-1, label_masks.shape[-1])

  preds = preds > pred_threshold
  intersection = np.logical_and(preds, label_masks)
  union = np.logical_or(preds, label_masks)
  iou_scores = np.sum(intersection, axis=0) / np.sum(union, axis=0)

  precision = np.sum(intersection, axis=0)/np.sum(preds, axis=0)
  recall = np.sum(intersection, axis=0)/np.sum(label_masks, axis=0)

  return iou_scores, precision, recall

"""
Performs one training step over a batch.
Passes one batch of images through the model, and backprops the gradients.
"""
@tf.function
def train_step(model, loss_function, train_loss, optimizer, images, labels):
  with tf.GradientTape() as tape:
    preds = model(images)
    loss = loss_function(labels, preds)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  loss = train_loss(loss)

  return loss, preds

"""
Performs one validation step over a batch.
"""
@tf.function
def val_step(model, loss_function, val_loss, optimizer, images, labels):
  preds = model(images)
  loss = loss_function(labels, preds)

  loss = val_loss(loss)

  return loss, preds


def passed_arguments():
  parser = argparse.ArgumentParser(description="Script to train an Image Segmentation model.")
  parser.add_argument('--data_path',
                      type=str,
                      required=True,
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--classes_path',\
                      type=str,
                      default='./classes.json',
                      help='Path to directory where extracted dataset is stored.')
  parser.add_argument('--epochs',
                      type=int,
                      default=100,
                      help='Number of epochs to train the model.')
  parser.add_argument('--batch_size',
                      type=int,
                      default=32,
                      help='Size of batches to feed into model.')
  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = passed_arguments()

  epochs = args.epochs
  batch_size = args.batch_size
  dataset = ImSeg_Dataset(data_path=args.data_path, classes_path=args.classes_path)
  img_size = dataset.image_size
  num_classes = len(dataset.seg_classes)

  # Number of training/validation samples and number of batches
  num_train, num_val = dataset.data_sizes[0], dataset.data_sizes[1]
  num_train_batches, num_val_batches = num_train//batch_size, num_val//batch_size

  ## BEGIN: REFACTOR THIS CODE FOR BETTER MODEL LOADING
  backbone_model = backbone.resnet50()

  model = network.create_refine_net(backbone_model, [['layer3', 'layer4']], num_classes, input_shape=img_size)
  ## END: REFACTOR CODE 

  # Loss and optimizer
  loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  val_loss = tf.keras.metrics.Mean(name='val_loss')

  # Alternate between training and validation epochs.
  for epoch in range(epochs):
    train_indices, val_indices = list(range(num_train)), list(range(num_val))
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    for batch in range(num_train_batches):
      img_input, label_masks =\
        dataset.get_batch(train_indices[batch*batch_size : (batch+1)*batch_size], "train")
      
      loss, preds = train_step(model, loss_function, train_loss, optimizer, img_input, label_masks)


    for batch in range(num_val_batches):
      img_input, label_masks =\
        dataset.get_batch(val_indices[batch*batch_size : (batch+1)*batch_size], "val")
      
      loss, preds = val_step(model, loss_function, val_loss, optimizer, img_input, label_masks)

      # Get metrics
      preds = preds.numpy()
      ious, prec, recall = calculate_iou_prec_recall(preds, label_masks, pred_threshold=0.5)


  





