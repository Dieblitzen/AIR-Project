import tensorflow as tf
import numpy as np
#launch session to connect to C++ computation power
sess = tf.InteractiveSession()

# IS STRIDE BEING HANDLED CORRECTLY?

""" Initializes weights with a slightly positive bias."""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

""" Initializes bias with a slightly positive bias."""
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Handle stride size -> stride size is one, so we slide filter over input one pixel at a time."""
def conv2d(input, filter_size, in_channels, out_channels, stride=1, padding="SAME"):
    return tf.nn.conv2d(input=input, 
          filter=[filter_size, filter_size, in_channels, out_channels],
          strides = [1, stride, stride, stride], padding = "SAME")


#Initialize expected input for images
x = tf.placeholder(tf.float32, [None, 228, 228, 3])
#Initialize expected output for per-pixel bounding boxes
y_box = tf.placeholder(tf.float32, [None, 228, 228, 6])
#Initialize expected output for per-pixel labels
y_class = tf.placeholder(tf.float32, [None, 228, 228, 1])

# two convolutional layers, 3x3, 32 filters
conv1 = tf.nn.relu(conv2d(input=x, filter_size=3, in_channels=3, out_channels=32) + b_conv1)
conv2 = tf.nn.relu(conv2d(input=conv1, filter_size=3, in_channels=32, out_channels=32) + b_conv1)

# resnet block 1
block1_shortcut = conv2
block1_shortcut_proj = tf.nn.relu(conv2d(input=block1_shortcut, filter_size=1, in_channels=32, out_channels=96) + b_conv1)

block1_1 = tf.nn.relu(conv2d(input=conv2, filter_size=3, in_channels=32, out_channels=24, stride=2) + b_conv1)
block1_2 = tf.nn.relu(conv2d(input=block1_1, filter_size=3, in_channels=24, out_channels=24) + b_conv1)
block1_3 = tf.nn.relu(conv2d(input=block1_2, filter_size=3, in_channels=24, out_channels=96) + b_conv1)

block1_out = block1_3 + block1_shortcut_proj

# resnet block 2 [Compressed from original version for now]
block2_shortcut = block1_out
block2_shortcut_proj = tf.nn.relu(conv2d(input=block2_shortcut, filter_size=1, in_channels=96, out_channels=192) + b_conv1)

block2_1 = tf.nn.relu(conv2d(input=block1_out, filter_size=3, in_channels=96, out_channels=48, stride=2) + b_conv1)
block2_2 = tf.nn.relu(conv2d(input=block2_1, filter_size=3, in_channels=48, out_channels=48) + b_conv1)
block2_3 = tf.nn.relu(conv2d(input=block2_2, filter_size=3, in_channels=48, out_channels=192) + b_conv1)

block2_out = block2_3 + block2_shortcut_proj

# skip connection from this output
skip_block2 = block2_out

# resnet block 3 [Compressed from original version for now]
block3_shortcut = block2_out
block3_shortcut_proj = tf.nn.relu(conv2d(input=block3_shortcut, filter_size=1, in_channels=192, out_channels=256) + b_conv1)

block3_1 = tf.nn.relu(conv2d(input=block2_out, filter_size=3, in_channels=192, out_channels=64, stride=2) + b_conv1)
block3_2 = tf.nn.relu(conv2d(input=block3_1, filter_size=3, in_channels=64, out_channels=64) + b_conv1)
block3_3 = tf.nn.relu(conv2d(input=block3_2, filter_size=3, in_channels=64, out_channels=256) + b_conv1)

block3_out = block3_3 + block3_shortcut_proj

# skip connection from this output
skip_block3 = block3_out

# resnet block 4
block4_shortcut = block3_out
block4_shortcut_proj = tf.nn.relu(conv2d(input=block4_shortcut, filter_size=1, in_channels=256, out_channels=384) + b_conv1)

block4_1 = tf.nn.relu(conv2d(input=block3_out, filter_size=3, in_channels=256, out_channels=96, stride=2) + b_conv1)
block4_2 = tf.nn.relu(conv2d(input=block4_1, filter_size=3, in_channels=96, out_channels=96) + b_conv1)
block4_3 = tf.nn.relu(conv2d(input=block4_2, filter_size=3, in_channels=96, out_channels=384) + b_conv1)

block4_out = block4_3 + block4_shortcut_proj

# one convolutional layer, 1x1, 196 filters
prep_upsampling = tf.nn.relu(conv2d(input=block4_out, filter_size=1, in_channels=384, out_channels=196, stride=1) + b_conv1)

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
