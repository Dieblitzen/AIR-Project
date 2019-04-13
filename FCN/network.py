## Network for FCN 

import numpy as np
import tensorflow as tf

"""
conv_layer(input, filter_shape, stride) creates a new convolutional layer and 
returns the convolution of the input. 
It uses weights/biases created based on filter_shape, stride and padding

Requires:
  input: the input Tensor
  filter_shape: [filter_height, filter_width, depth_of_input, n_filters]
  stride: [batch=1, horizontal_stride, vertical_stride, depth_of_convolution=1]
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
"""
def conv_layer(input_t, filter_shape, stride=[1,1,1,1], padding='SAME'):
  # Have to define weights when using tf.nn.conv2d
  weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
  biases = tf.Variable(tf.zeros([filter_shape[3]]))

  return tf.nn.conv2d(input_t, weights, stride, padding) + biases

"""
batch_norm(x) batch normalises the input tensor x
Requires:
  x: input Tensor
"""
def batch_norm(x):
  # Calculates the the mean and variance of x
  mean, variance = tf.nn.moments(x, axes=[0])
  return tf.nn.batch_normalization(x, mean, variance, None, None, 0.001)


"""
resnet_block(..) performs the (non-downsampled) convolutions on the input tensor
Requires:
  input_t: the input tensor
  filter_shape: [filter_height, filter_width, depth_of_input, n_filters]
  stride: [batch=1, horizontal_stride, vertical_stride, depth_of_convolution=1]
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
  n_layers: the number of convolutional layers within the the block
"""
def resnet_block(input_t, filter_shape, stride=[1,1,1,1], padding='SAME', n_layers=2):

  identity = input_t
  x = input_t

  for _ in range(n_layers):
    x = batch_norm(x)
    x = tf.nn.relu(x)
    x = conv_layer(x, filter_shape, stride, padding)
  
  return x + identity


"""
resnet_block_bottleneck(..) performs the (downsampled) convolutions on the input tensor
Downsamples in the first convolution, assumes depth doubles from previous feature map
Requires:
  input_t: the input tensor
  filter_shape: [filter_height, filter_width, depth_of_input, n_filters]
                depth_of_input should be with respect to output in this case.
  stride: [batch=1, horizontal_stride, vertical_stride, depth_of_convolution=1]
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
  n_layers: the number of convolutional layers within the the block
"""
def resnet_block_bottleneck(input_t, filter_shape, stride=[1,1,1,1], padding='SAME', n_layers=2):
  identity = conv_layer(input_t, [1,1,filter_shape[2], filter_shape[3]], [1,2,2,1], padding='SAME')
  x = input_t

  # Downsampled
  x = batch_norm(x)
  x = tf.nn.relu(x)
  x = conv_layer(
    x, [filter_shape[0], filter_shape[1], filter_shape[2]/2, filter_shape[3]], [1,2,2,1], padding)

  for _ in range(1, n_layers):
    x = batch_norm(x)
    x = tf.nn.relu(x)
    x = conv_layer(x, filter_shape, stride, padding)
  
  return x + identity


