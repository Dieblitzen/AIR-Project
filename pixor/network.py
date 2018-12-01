import tensorflow as tf
import numpy as np
#launch session to connect to C++ computation power
sess = tf.InteractiveSession()

""" Initializes weights with a slightly positive bias."""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

""" Initializes bias with a slightly positive bias."""
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Handle stride size -> stride size is one, so we slide filter over input one pixel at a time."""
def conv2d(input, filter_height, filter_width, in_channels, out_channels, stride=1, padding="SAME"):
    return tf.nn.conv2d(input=input, 
          filter=[filter_height, filter_width, in_channels, out_channels],
          strides = [1, stride, stride, stride], padding = "SAME")


#Initialize expected input for images
x = tf.placeholder(tf.float32, [None, 228, 228, 3])
#Initialize expected output for per-pixel bounding boxes
y_box = tf.placeholder(tf.float32, [None, 228, 228, 6])
#Initialize expected output for per-pixel labels
y_class = tf.placeholder(tf.float32, [None, 228, 228, 1])

# two convolutional layers, 3x3, 32 filters

# resnet block 2

# resnet block 3

  # skip connection from this output

# resnet block 4

  # skip connection from this output

# resnet block 5

# one convolutional layer, 1x1, 196 filters

# upsample 6, 128 filters , x2 

  # postprocessing to add skip connection after upsample 6
    # [1x1, 128 channel convolution on skip ;; then add]

# upsample 7, 96 filters, x2

  # postprocessing to add skip connection after upsample 7
    # [1x1, 96 channel convolution on skip ;; then add]

## HEADER NETWORK

# four convolutional layers, 3x3, 96 filters

  # one convolutional layer, 3x3, 1 filter

  # one convolutional layer, 3x3, 6 filters
