import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from data_extract import extract_data
from sklearn.utils import shuffle
#launch session to connect to C++ computation power
saver = tf.train.Saver()
sess = tf.InteractiveSession()

# LACKING MORE SKIP CONNECTIONS

BATCH_SIZE=32

""" Initializes weights with a slightly positive bias."""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

""" Initializes bias with a slightly positive bias."""
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

""" Standard convolutional layer."""
def conv2d(input, filter_size, in_channels, out_channels, stride=1, padding="SAME"):
  weights = weight_variable([filter_size, filter_size, in_channels, out_channels])
  return tf.nn.conv2d(input=input, 
        filter=weights,
        strides=[1, stride, stride, 1], padding=padding) + bias_variable([out_channels])

""" Standard transposed convolutional layer."""
def conv2d_transpose(input, filter_size, out_channels, stride, activation="None"):
  return tf.layers.conv2d_transpose(inputs=input, filters=out_channels,
         kernel_size=filter_size, strides=stride, padding='same', activation=activation)


#Initialize expected input for images
x = tf.placeholder(tf.float32, shape=(None, 228, 228, 3))
#Initialize holder for per-pixel bounding boxes
y_box = tf.placeholder(tf.float32, shape=(None, 228, 228, 6))
#Initialize holder for per-pixel labels
y_class = tf.placeholder(tf.float32, shape=(None, 228, 228, 1))

# two convolutional layers, 3x3, 32 filters
conv1 = tf.nn.relu(conv2d(input=x, filter_size=3, in_channels=3, out_channels=32))
conv2 = tf.nn.relu(conv2d(input=conv1, filter_size=3, in_channels=32, out_channels=32))

# resnet block 1
block1_shortcut = conv2
block1_shortcut_proj = tf.nn.relu(conv2d(input=block1_shortcut, filter_size=1, in_channels=32, out_channels=96, stride=2))

block1_1 = tf.nn.relu(conv2d(input=conv2, filter_size=3, in_channels=32, out_channels=24, stride=2))
block1_2 = tf.nn.relu(conv2d(input=block1_1, filter_size=3, in_channels=24, out_channels=24))
block1_3 = tf.nn.relu(conv2d(input=block1_2, filter_size=3, in_channels=24, out_channels=96))

block1_out = block1_3 + block1_shortcut_proj

# resnet block 2 [Compressed from original version for now]
block2_shortcut = block1_out
block2_shortcut_proj = tf.nn.relu(conv2d(input=block2_shortcut, filter_size=1, in_channels=96, out_channels=192, stride=2))

block2_1 = tf.nn.relu(conv2d(input=block1_out, filter_size=3, in_channels=96, out_channels=48, stride=2))
block2_2 = tf.nn.relu(conv2d(input=block2_1, filter_size=3, in_channels=48, out_channels=48))
block2_3 = tf.nn.relu(conv2d(input=block2_2, filter_size=3, in_channels=48, out_channels=192))

block2_out = block2_3 + block2_shortcut_proj

# skip connection from this output
skip_block2 = block2_out

# resnet block 3 [Compressed from original version for now]
block3_shortcut = block2_out
block3_shortcut_proj = tf.nn.relu(conv2d(input=block3_shortcut, filter_size=1, in_channels=192, out_channels=256, stride=2))

block3_1 = tf.nn.relu(conv2d(input=block2_out, filter_size=3, in_channels=192, out_channels=64, stride=2))
block3_2 = tf.nn.relu(conv2d(input=block3_1, filter_size=3, in_channels=64, out_channels=64))
block3_3 = tf.nn.relu(conv2d(input=block3_2, filter_size=3, in_channels=64, out_channels=256))

block3_out = block3_3 + block3_shortcut_proj

# skip connection from this output
skip_block3 = block3_out

# resnet block 4
block4_shortcut = block3_out
block4_shortcut_proj = tf.nn.relu(conv2d(input=block4_shortcut, filter_size=1, in_channels=256, out_channels=384, stride=2))

block4_1 = tf.nn.relu(conv2d(input=block3_out, filter_size=3, in_channels=256, out_channels=96, stride=2))
block4_2 = tf.nn.relu(conv2d(input=block4_1, filter_size=3, in_channels=96, out_channels=96))
block4_3 = tf.nn.relu(conv2d(input=block4_2, filter_size=3, in_channels=96, out_channels=384))

block4_out = block4_3 + block4_shortcut_proj

# one convolutional layer, 1x1, 196 filters
prep_upsampling = tf.nn.relu(conv2d(input=block4_out, filter_size=1, in_channels=384, out_channels=196, stride=1))

# upsample 6, 128 filters , x2 
upsample1 = conv2d_transpose(input=prep_upsampling, filter_size=3, out_channels=128, stride=2, activation=tf.nn.relu)
# fix dimensions for addition
upsample1 = upsample1[:, :-1, :-1, :]

# postprocessing to add skip connection after upsample 6
# [1x1, 128 channel convolution on skip ;; then add]
processed_skip_block3 = tf.nn.relu(conv2d(input=skip_block3, filter_size=1, in_channels=256, out_channels=128, stride=1))
skipped_upsample1 = upsample1 + processed_skip_block3

# upsample 7, 96 filters, x2
upsample2 = conv2d_transpose(input=skipped_upsample1, filter_size=3, out_channels=96, stride=2, activation=tf.nn.relu)
upsample2 = upsample2[:, :-1,  :-1, :]

# postprocessing to add skip connection after upsample 7
# [1x1, 96 channel convolution on skip ;; then add]
processed_skip_block2 = tf.nn.relu(conv2d(input=skip_block2, filter_size=1, in_channels=192, out_channels=96, stride=1))
skipped_upsample2 = upsample2 + processed_skip_block2

# PLACEHOLDER UPSAMPLING
temp_final_upsample = conv2d_transpose(input=skipped_upsample2, filter_size=3, out_channels=96, stride=4, activation=tf.nn.relu)

## HEADER NETWORK

# four convolutional layers, 3x3, 96 filters
header1 = tf.nn.relu(conv2d(input=temp_final_upsample, filter_size=3, in_channels=96, out_channels=96, stride=1))
header2 = tf.nn.relu(conv2d(input=header1, filter_size=3, in_channels=96, out_channels=96, stride=1))
header3 = tf.nn.relu(conv2d(input=header2, filter_size=3, in_channels=96, out_channels=96, stride=1))
header4 = tf.nn.relu(conv2d(input=header3, filter_size=3, in_channels=96, out_channels=96, stride=1))

# one convolutional layer, 3x3, 1 filter
output_box = tf.nn.relu(conv2d(input=header4, filter_size=3, in_channels=96, out_channels=1, stride=1))

# one convolutional layer, 3x3, 6 filters
output_class = tf.nn.relu(conv2d(input=header4, filter_size=3, in_channels=96, out_channels=6, stride=1))


# DEFINING LOSS

""" If absolute value of difference < 1 -> 0.5 * (abs(difference))^2. 
Otherwise, abs(difference) - 0.5. """
def smooth_L1(box_labels, box_preds, class_labels):
  difference = tf.subtract(box_preds, box_labels)
  abs_difference = tf.abs(difference)
  result = tf.where(abs_difference < 1, 0.5 * abs_difference ** 2, abs_difference - 0.5)
  # only compute bbox loss over positive ground truth boxes
  cleaned_result = tf.boolean_mask(result, class_labels.flatten())
  return tf.reduce_sum(cleaned_result)

class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_class, logits=output_class))
box_loss = smooth_L1(box_labels=y_box, box_preds=output_box, class_labels=y_class)
pixor_loss = class_loss + box_loss

#A step to minimize our cost function
train_step = tf.train.AdamOptimizer(1e-4).minimize(pixor_loss)

# RUN THINGS

with tf.Session() as sess:
  # load in data
  tile_list = extract_data("../tiles3.pkl")

  datapoints = None

  train_data = None
  train_classlabels = None
  train_boxlabels = None

  val_data = None
  val_classlabels = None
  val_boxlabels = None

  tile_array = np.asarray(tile_list)
  print(tile_array)

  #initialize everything
  sess.run(tf.global_variables_initializer())
  num_epochs = 10
  
  lowest_val_loss = np.inf
  for epoch in range(num_epochs):

    # shuffle data to randomize order of network exposure
    train_data, train_classlabels, train_boxlabels = 
          shuffle(train_data, train_classlabels,train_boxlabels)

    num_batches = train_data.shape[0] // BATCH_SIZE
    for batch_number in range(0, num_batches):
      start_idx = batch_number * BATCH_SIZE
      end_idx = start_idx + BATCH_SIZE

      # train on the batch
      train_step.run(feed_dict = 
        {x: train_data[start_idx: end_idx], 
        y_box: train_boxlabels[start_idx: end_idx], 
        y_class: train_classlabels[start_idx: end_idx], keep_prob: 0.5})
  
    # at each epoch, print training and validation loss
    train_loss = pixor_loss.eval(feed_dict = {x: train_data, 
        y_box: train_boxlabels, y_class: train_classlabels, keep_prob: 1.0})
    val_loss = pixor_loss.eval(feed_dict = {x: val_data, 
        y_box: val_boxlabels, y_class: val_classlabels, keep_prob: 1.0})
    print('epoch %d, training loss %g' % (epoch, train_loss))
    print('epoch %d, validation loss %g' % (epoch, val_loss))

    # checkpoint model if best so far
    if val_loss < lowest_val_loss:
      saver.save(sess, 'my-model', global_step=epoch)
