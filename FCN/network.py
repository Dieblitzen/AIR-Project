## Network for FCN 

import numpy as np
import tensorflow as tf
import ImSeg_Dataset.ImSeg_Dataset as Data
import os

IM_SIZE = [224,224,3]


"""
conv_layer(input, filter_shape, stride) creates a new convolutional layer and 
returns the convolution of the input. 
It uses weights/biases created based on filter_shape, stride and padding

Requires:
  input: the input Tensor [batch, in_height, in_width, in_channels]
  filter_shape: [filter_height, filter_width, in_channels, out_channels]
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
  identity = conv_layer(input_t, [1,1,filter_shape[2]/2, filter_shape[3]], [1,2,2,1], padding)
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

## Define Upsampling
"""
deconv_layer(input, filter_shape, stride) creates a new transpose convolutional layer 
and returns the transpose convolution of the input. 
It uses weights/biases created based on filter_shape, stride and padding

Requires:
  input: the input Tensor [batch, height, width, in_channels]
  filter_shape: [filter_height, filter_width, output_channels, input_channels]
  output_shape: [batch, height, width, channels]
  stride: [batch=1, horizontal_stride, vertical_stride, depth_of_convolution=1]
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
"""
def deconv_layer(input_t, filter_shape, output_shape, stride=[1,2,2,1], padding='SAME'):
  # Have to define weights when using tf.nn.conv2d_transpose
  weights = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05))
  biases = tf.Variable(tf.zeros([filter_shape[2]]))
  return tf.nn.conv2d_transpose(input_t, weights, output_shape, stride, padding) + biases


## Define ResNet architecture

# Input and output image placeholders
# Shape is [None, IM_SIZE] where None indicates variable batch size
X = tf.placeholder(tf.float32, shape=[None] + IM_SIZE)
y = tf.placeholder(tf.float32, shape=[None] + IM_SIZE) 

# Downsampling /2
block_1 = conv_layer(X, [7,7,3,64], stride=[1,2,2,1]) # 2 downsampled

# Downsampling /2
# tf.nn.maxpool ksize is [batch, width, height, channels]. batch and channels is 1
# because we don't want to take the max over multiple examples or multiple channels.
block_1_pooled = tf.nn.max_pool(block_1, [1,3,3,1], [1,2,2,1], padding='SAME')

block_2 = resnet_block(block_1_pooled, [3,3,64,64])
block_3 = resnet_block(block_2, [3,3,64,64])
block_4 = resnet_block(block_3, [3,3,64,64]) # 4 downsampled

# Downsampling /2
block_5 = resnet_block_bottleneck(block_4, [3,3,128,128])
block_6 = resnet_block(block_5, [3,3,128,128])
block_7 = resnet_block(block_6, [3,3,128,128])
block_8 = resnet_block(block_7, [3,3,128,128]) # 8 downsampled

# Downsampling /2
block_9 = resnet_block_bottleneck(block_8, [3,3,256,256])
block_10 = resnet_block(block_9, [3,3,256,256])
block_11 = resnet_block(block_10, [3,3,256,256])
block_12 = resnet_block(block_11, [3,3,256,256])
block_13 = resnet_block(block_12, [3,3,256,256])
block_14 = resnet_block(block_13, [3,3,256,256]) # 16 downsampled

# Downsampling / 2
block_15 = resnet_block_bottleneck(block_14, [3,3,512,512])
block_16 = resnet_block(block_15, [3,3,512,512])
block_17 = resnet_block(block_16, [3,3,512,512]) # 32 downsampled

# At size 7x7 at this point.

## FCN-8
upsampled_32 = deconv_layer(block_17, [3,3,256,512], [1,14,14,256], [1,2,2,1])
pool_4_and_5 = upsampled_32 + block_14

upsampled_32_16 = deconv_layer(pool_4_and_5, [3,3,128,256], [1,28,28,128], [1,2,2,1]) 
pool_3_and_4 = upsampled_32_16 + block_8

fcn8 = deconv_layer(pool_3_and_4, [3,3,1,128], [1,224,224,1], [1,8,8,1])


## Building the model
## Global variables

learning_rate = 0.001
num_epochs = 300
batch_size = 32

# Defining Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fcn8)
loss = tf.reduce_mean(loss)

# Use an Adam optimizer to train network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializer for global variables, to be run at the beginning of a session
init = tf.global_variables_initializer()


if __name__ == "__main__":
  
  ## Get the data
  data = Data('./data_path_white_plains')
  if not os.path.isdir('./data_path_white_plains/im_seg'):
    data = Data.build_dataset()

  

  # Begin session
  with tf.Session() as sess: 
    sess.run(init)

    # for each epoch: 
    # . for each batch:
    # . . create batch
    # . . run backprop/optimizer by feeding in batch
    # . . find loss for the batch
    # . print loss for the epoch
    # . print testing accuracy for each epoch
    
    # Number of training samples and number of batches
    num_train = data.data_sizes[0]
    num_batches = num_train//batch_size

    # Validation size
    num_val = data.data_sizes[1]
    
    # Training
    for epoch in range(num_epochs):
      indices = list(range(num_train))
      np.random.shuffle(indices)
      # Track epoch loss
      epoch_loss = 0
      for batch in range(num_batches):
        
        # Get the batch
        X_batch, y_batch = data.get_batch(indices[batch*batch_size : (batch+1)*batch_size], path="train")

        ## Resize images (unimplemented)

        # Since it is a dictionary, X (defined above) gets the batch of images X_batch (same for y)
        _, batch_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, y:y_batch})

        epoch_loss += batch_loss
      
      ## TODO: Save weights at each epoch.
      X_val_batch, y_val_batch = data.get_batch([i for i in range(num_val)], path="val")
      preds = sess.run([fcn8], feed_dict={X:X_val_batch, y:y_val_batch})

      print(f"Loss at epoch {epoch+1}: {epoch_loss}")











