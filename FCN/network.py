## Network for FCN 

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import scipy.misc
from ImSeg_Dataset import ImSeg_Dataset as Data
import os
from PIL import Image
import logging

## Global variables
IM_SIZE = [224,224,3]
LABEL_SIZE = [224,224,1]

learning_rate = 0.0001
num_epochs = 300
batch_size = 32
pred_threshold = 0.5


## =============================================================================================
## Basic CNN Operations
## =============================================================================================

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

## =============================================================================================
## Define Res-Net Operations
## =============================================================================================

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
  identity = conv_layer(input_t, [1,1,filter_shape[2]//2, filter_shape[3]], [1,2,2,1], padding)
  x = input_t

  # Downsampled
  x = batch_norm(x)
  x = tf.nn.relu(x)
  x = conv_layer(
    x, [filter_shape[0], filter_shape[1], filter_shape[2]//2, filter_shape[3]], [1,2,2,1], padding)

  for _ in range(1, n_layers):
    x = batch_norm(x)
    x = tf.nn.relu(x)
    x = conv_layer(x, filter_shape, stride, padding)
  
  return x + identity

## =============================================================================================
## Define RefineNet Operations
## =============================================================================================

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

"""
Residual Convolution Unit
- Essentially a resnet block without the batch norm.
- Uses 3x3 convolutions (maintains dimensions of input.)
Requires:
  input_t: the input tensor
  filter_shape: [filter_height, filter_width, depth_of_input, n_filters]
  stride: [batch=1, horizontal_stride, vertical_stride, depth_of_convolution=1]
  padding: string of 'SAME' (1/stride * input_size) or 'VALID' (no padding)
  n_layers: the number of convolutional layers within the the block
"""
def rcu_block(input_t, n_layers=2):
  identity = input_t
  x = input_t
  
  for _ in range(n_layers):
    x = tf.nn.relu(x)
    x = conv_layer(x, [3, 3, int(x.get_shape()[3]), int(x.get_shape()[3]) ]) 
  
  return x + identity

"""
Multi-resolution Fusion.
- Fuses inputs into high-res feature map. First applies 3x3 convolutions to create feature maps
  of same depth dimension (smallest depth of channels among inputs).
- Upsamples the smaller feature maps to largest resolution of inputs, then sums them all.
Requires:
  input_tensors: a list of tensors (usually different dimensions) being inputted to the block.
"""
def mrf_block(input_tensors):
  
  # Convolve input tensors using 3x3 filters. 
  # All output tensors will have same channel depth (# of channels)
  convolved = []
  smallest_depth = min(input_tensors, key=lambda t: int(t.get_shape()[3]) )
  smallest_depth = int(smallest_depth.get_shape()[3])
  for t in input_tensors:
    x = conv_layer(t, [3, 3, int(t.get_shape()[3]), smallest_depth] )
    convolved.append(x)
  
  # Upsample the convolutions to the largest input tensor resolution.
  # Assuming width and height dimensions are the same for each tensor.
  up_sampled = []
  largest_res = max(input_tensors, key=lambda t: int(t.get_shape()[1]) )
  largest_res = int(largest_res.get_shape()[1])
  for t in convolved:
    old_res = int(t.get_shape()[1]) # The width/height of the old tensor.
    x = deconv_layer(t, [3, 3, smallest_depth, smallest_depth], \
                        [batch_size, largest_res, largest_res, smallest_depth], \
                        stride=[1, largest_res//old_res, largest_res//old_res, 1])
    up_sampled.append(x)
  
  # Sum them all up
  return sum(up_sampled)


"""
Chained Residual Pooling.
- Chain of multiple pooling blocks, each consisting of one max-pooling layer
  and one convolutional layer. Kernel size for pooling is 5.
- Output feature maps of pooling blocks are summed with identity mappings.
- Maintains the dimensions of the input.
Requires:
  input_t is the output of the mrf_block. 
"""
def crp_block(input_t, n_pool_blocks=2):
  result = input_t
  x = tf.nn.relu(input_t)

  for _ in range(n_pool_blocks):
    x = tf.nn.max_pool(x, [1,5,5,1], [1,1,1,1], padding="SAME")
    x = conv_layer(x, [3, 3, int(x.get_shape()[3]), int(x.get_shape()[3]) ] )
    result = result + x

  return result


"""
RefineNet block.
- Applies Residual Convolution Units twice to each input tensor
- Fuses them together with Multi-Resolution Fusion.
- Applies Chained Residual Pooling
- Applies Residual Convolution Unit once one last time.
Requires:
  input_tensors: A list of tensors to pass through the refine net.
"""
def refine_net_block(input_tensors):
  # Apply Residual Convolution Units twice to each input tensor
  rcu = []
  for t in input_tensors:
    x = rcu_block(rcu_block(t))
    rcu.append(x)
  
  # Apply Multi-Resolution Fusion
  mrf = mrf_block(rcu)

  # Apply Chained Residual Pooling
  crp = crp_block(mrf)

  # Apply Residual Convolution Unit one last time
  return rcu_block(crp)


## =============================================================================================
## Define Res-Net architecture
## =============================================================================================

# Input and output image placeholders
# Shape is [None, IM_SIZE] where None indicates variable batch size
X = tf.placeholder(tf.float32, shape=[None] + IM_SIZE)
y = tf.placeholder(tf.float32, shape=[None] + LABEL_SIZE) 

# Downsampling /2
block_1 = conv_layer(X, [7,7,3,64], stride=[1,2,2,1]) # 1/2 downsampled

# Downsampling /2
# tf.nn.maxpool ksize is [batch, width, height, channels]. batch and channels is 1
# because we don't want to take the max over multiple examples or multiple channels.
block_1_pooled = tf.nn.max_pool(block_1, [1,3,3,1], [1,2,2,1], padding='SAME')

block_2 = resnet_block(block_1_pooled, [3,3,64,64])
block_3 = resnet_block(block_2, [3,3,64,64])
block_4 = resnet_block(block_3, [3,3,64,64]) # 1/4 downsampled

# Downsampling /2
block_5 = resnet_block_bottleneck(block_4, [3,3,128,128])
block_6 = resnet_block(block_5, [3,3,128,128])
block_7 = resnet_block(block_6, [3,3,128,128])
block_8 = resnet_block(block_7, [3,3,128,128]) # 1/8 downsampled

# Downsampling /2
block_9 = resnet_block_bottleneck(block_8, [3,3,256,256])
block_10 = resnet_block(block_9, [3,3,256,256])
block_11 = resnet_block(block_10, [3,3,256,256])
block_12 = resnet_block(block_11, [3,3,256,256])
block_13 = resnet_block(block_12, [3,3,256,256])
block_14 = resnet_block(block_13, [3,3,256,256]) # 1/16 downsampled

# Downsampling / 2
block_15 = resnet_block_bottleneck(block_14, [3,3,512,512])
block_16 = resnet_block(block_15, [3,3,512,512])
block_17 = resnet_block(block_16, [3,3,512,512]) # 1/32 downsampled

# At size 7x7 at this point.

## =============================================================================================
## Apply FCN-8
## =============================================================================================
# upsampled_32 = deconv_layer(block_17, [3,3,256,512], [batch_size,14,14,256], [1,2,2,1])
# pool_4_and_5 = upsampled_32 + block_14

# upsampled_32_16 = deconv_layer(pool_4_and_5, [3,3,128,256], [batch_size,28,28,128], [1,2,2,1]) 
# pool_3_and_4 = upsampled_32_16 + block_8

# fcn8 = deconv_layer(pool_3_and_4, [3,3,1,128], [batch_size,224,224,1], [1,8,8,1])


## =============================================================================================
## Apply Refine-Net 
## =============================================================================================

# Refine Net returns result 1/4 the size of input. Still need to upsample 4 times.
# Expect the depth of the refine net output to be 64, since that is depth of #4 downsampled.
upsampled = refine_net_block([block_17, block_14, block_8, block_4])
result = deconv_layer(upsampled, [3,3,1,64], [batch_size,224,224,1], [1,4,4,1])


## =============================================================================================
## Building the model
## =============================================================================================

# Defining Loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=result)
loss = tf.reduce_mean(loss)

# Use an Adam optimizer to train network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializer for global variables, to be run at the beginning of a session
init = tf.global_variables_initializer()


if __name__ == "__main__":

  ## Set up the logger
  logging.basicConfig(filename="ImSegEval.log")
  
  ## Get the data
  data = Data('../data_path_white_plains_224')
  if not os.path.isdir('../data_path_white_plains_224/im_seg'):
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
    num_train_batches = num_train//batch_size

    # Number of validation samples and number of batches
    num_val = data.data_sizes[1]
    num_val_batches = num_val//batch_size
    
    # Training
    for epoch in range(num_epochs):

      # Shuffle indices of training image to randomise batch selection
      train_indices = list(range(num_train))
      np.random.shuffle(train_indices)

      val_indices = list(range(num_val))
      np.random.shuffle(val_indices)

      # Track epoch loss and IoU
      epoch_train_loss = 0
      epoch_val_loss = 0
      epoch_IoU = 0

      ## Training the model and recording training loss
      for batch in range(num_train_batches):
        
        # Get the batch
        X_batch, y_batch = data.get_batch(train_indices[batch*batch_size : (batch+1)*batch_size], "train")

        ## TODO: Resize images (unimplemented)

        # Since it is a dictionary, X (defined above) gets the batch of images X_batch (same for y)
        _, train_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, y:y_batch})
        
        # Record the training loss
        epoch_train_loss += train_loss
      
      ## Recording validation loss
      for batch in range(num_val_batches):

        # Get the batch
        X_batch, y_batch = data.get_batch(val_indices[batch*batch_size : (batch+1)*batch_size], "val")
        
        ## TODO: Resize images (unimplemented)

        # Get the predictions
        preds, val_loss = sess.run([result, loss], feed_dict={X:X_batch, y:y_batch})
        
        # Record the validation loss
        epoch_val_loss += val_loss

        # Calculate IoU for entire image.
        preds = preds > pred_threshold
        intersection = np.logical_and(preds, y_batch)
        union = np.logical_or(preds, y_batch)
        iou_score = np.sum(intersection) / np.sum(union)
        epoch_IoU += iou_score

        if (epoch+1) % 100 == 0:
          data.save_preds(val_indices[batch*batch_size : (batch+1)*batch_size], preds, image_dir="val")
          

      ## Average the loss, and display the result (multiply by 10 to make it readable)
      epoch_train_loss = epoch_train_loss/num_train_batches * 10
      epoch_val_loss = epoch_val_loss/num_val_batches * 10
      epoch_IoU = epoch_IoU / num_val_batches

      logging.info("Epoch: " + str(epoch+1) + ", Training Loss: " + str(epoch_train_loss))
      logging.info("Epoch: " + str(epoch+1) + ", Validation Loss: " + str(epoch_val_loss))
      logging.info("Epoch: " + str(epoch+1) + ", Epoch IoU: " + str(epoch_IoU))

      print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}")
      print(f"                 Validation Loss: {epoch_val_loss}")
      print(f"                 IoU score: {epoch_IoU}")

      ## TODO: Save weights with checkpoint files.

      # # Save predictions
      # if (epoch+1)%100 == 0:

      #   i = 0
      #   for pred in preds[0]:
      #     pred = np.squeeze(pred) #Drop the last dimesnion, which is anyways 1
      #     pred = np.array(pred > pred_threshold, dtype=np.int32)
      #     # Pass in pred here
      #     print(pred.shape)
      #     img = Image.fromarray(pred, mode="L")
      #     img.save(f'epoch{epoch}_{i}.png')
      #     i += 1












