import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import logging

logging.basicConfig(level=logging.INFO, filename="PIXOR_logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
sys.path.append("..")

#launch session to connect to C++ computation power
sess = tf.InteractiveSession()

# LACKING MORE SKIP CONNECTIONS

BATCH_SIZE=32

""" Standard transposed convolutional layer."""
def conv2d_transpose(input, filter_size, out_channels, stride, activation="None"):
  return tf.layers.conv2d_transpose(inputs=input, filters=out_channels,
         kernel_size=filter_size, strides=stride, padding='same', activation=activation)


def get_tile_and_label(index, base_path):
    """
    Method 2)
    Gets the tile and label associated with data index.

    Returns:
    (tile_array, dictionary_of_buildings)
    """

    # Open the jpeg image and save as numpy array
    im = Image.open(base_path + '/images/' + str(index) + '.jpg')
    im_arr = np.array(im)

    # Open the json file and parse into dictionary of index -> buildings pairs
    box_annotation = np.load(base_path + '/box_annotations/' + str(index) + '.npy')
    class_annotation = np.load(base_path + '/class_annotations/' + str(index) + '.npy')
    
    return im_arr, box_annotation, class_annotation


def get_batch(start_index, batch_size, batch_indices, base_path):
    """
    Method 3)
    Gets batch of tiles and labels associated with data start_index.

    Returns:
    [(tile_array, list_of_buildings), ...]
    """
    batch_images = np.zeros((batch_size, 228, 228, 3))
    batch_boxes = np.zeros((batch_size, 228, 228, 6))
    batch_classes = np.zeros((batch_size, 228, 228, 1))
    for i in range(start_index, start_index + batch_size):
      batch_images[i % batch_size], batch_boxes[i % batch_size], batch_classes[i % batch_size] = get_tile_and_label(batch_indices[i], base_path)
    
    return batch_images, batch_boxes, batch_classes


#Initialize expected input for images
x = tf.placeholder(tf.float32, shape=(None, 228, 228, 3), name='x')
#Initialize holder for per-pixel bounding boxes
y_box = tf.placeholder(tf.float32, shape=(None, 228, 228, 6), name='y_box')
#Initialize holder for per-pixel labels
y_class = tf.placeholder(tf.float32, shape=(None, 228, 228, 1), name='y_class')

# two convolutional layers, 3x3, 32 filters
conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)

# resnet block 1
block1_shortcut = conv2
block1_shortcut_proj = tf.layers.conv2d(inputs=block1_shortcut, filters=96, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)

block1_1 = tf.layers.conv2d(inputs=conv2, filters=24, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
block1_2 = tf.layers.conv2d(inputs=block1_1, filters=24, kernel_size=3, padding='same', activation=tf.nn.relu)
block1_3 = tf.layers.conv2d(inputs=block1_2, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)

block1_out = block1_3 + block1_shortcut_proj

# resnet block 2 [Compressed from original version for now]
block2_shortcut = block1_out
block2_shortcut_proj = tf.layers.conv2d(inputs=block2_shortcut, filters=192, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)

block2_1 = tf.layers.conv2d(inputs=block1_out, filters=48, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
block2_2 = tf.layers.conv2d(inputs=block2_1, filters=48, kernel_size=3, padding='same', activation=tf.nn.relu)
block2_3 = tf.layers.conv2d(inputs=block2_2, filters=192, kernel_size=3, padding='same', activation=tf.nn.relu)

block2_out = block2_3 + block2_shortcut_proj

# skip connection from this output
skip_block2 = block2_out

# resnet block 3 [Compressed from original version for now]
block3_shortcut = block2_out
block3_shortcut_proj = tf.layers.conv2d(inputs=block3_shortcut, filters=256, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)

block3_1 = tf.layers.conv2d(inputs=block2_out, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
block3_2 = tf.layers.conv2d(inputs=block3_1, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
block3_3 = tf.layers.conv2d(inputs=block3_2, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu)

block3_out = block3_3 + block3_shortcut_proj

# skip connection from this output
skip_block3 = block3_out

# resnet block 4
block4_shortcut = block3_out
block4_shortcut_proj = tf.layers.conv2d(inputs=block4_shortcut, filters=384, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)

block4_1 = tf.layers.conv2d(inputs=block3_out, filters=96, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
block4_2 = tf.layers.conv2d(inputs=block4_1, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
block4_3 = tf.layers.conv2d(inputs=block4_2, filters=384, kernel_size=3, padding='same', activation=tf.nn.relu)

block4_out = block4_3 + block4_shortcut_proj

# one convolutional layer, 1x1, 196 filters
prep_upsampling = tf.layers.conv2d(inputs=block4_out, filters=196, kernel_size=1, activation=tf.nn.relu)

# upsample 6, 128 filters , x2 
upsample1 = conv2d_transpose(input=prep_upsampling, filter_size=3, out_channels=128, stride=2, activation=tf.nn.relu)
# fix dimensions for addition
upsample1 = upsample1[:, :-1, :-1, :]

# postprocessing to add skip connection after upsample 6
# [1x1, 128 channel convolution on skip ;; then add]
processed_skip_block3 = tf.layers.conv2d(inputs=skip_block3, filters=128, kernel_size=1, padding='same', activation=tf.nn.relu)
skipped_upsample1 = upsample1 + processed_skip_block3

# upsample 7, 96 filters, x2
upsample2 = conv2d_transpose(input=skipped_upsample1, filter_size=3, out_channels=96, stride=2, activation=tf.nn.relu)
upsample2 = upsample2[:, :-1,  :-1, :]

# postprocessing to add skip connection after upsample 7
# [1x1, 96 channel convolution on skip ;; then add]
processed_skip_block2 = tf.layers.conv2d(inputs=skip_block2, filters=96, kernel_size=1, padding='same', activation=tf.nn.relu)
skipped_upsample2 = upsample2 + processed_skip_block2

# PLACEHOLDER UPSAMPLING
temp_final_upsample = conv2d_transpose(input=skipped_upsample2, filter_size=3, out_channels=96, stride=4, activation=tf.nn.relu)

## HEADER NETWORK

# four convolutional layers, 3x3, 96 filters
header1 = tf.layers.conv2d(inputs=temp_final_upsample, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
header2 = tf.layers.conv2d(inputs=header1, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
header3 = tf.layers.conv2d(inputs=header2, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
header4 = tf.layers.conv2d(inputs=header3, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)

# one convolutional layer, 3x3, 1 filter
output_class = tf.layers.conv2d(inputs=header4, filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid, name='output_class')

# one convolutional layer, 3x3, 6 filters
output_box = tf.layers.conv2d(inputs=header4, filters=6, kernel_size=3, padding='same', name='output_box')


## DEFINING LOSS

""" If absolute value of difference < 1 -> 0.5 * (abs(difference))^2. 
Otherwise, abs(difference) - 0.5. """
def smooth_L1(box_labels, box_preds, class_labels):
  difference = tf.subtract(box_preds, box_labels)
  # ones = tf.fill([BATCH_SIZE, 228, 228, 6], 1.)
  # halves = tf.fill([BATCH_SIZE, 228, 228, 6], 0.5)
  comp = tf.less(tf.abs(difference), 1)
  result = tf.where(comp, tf.multiply(0.5, tf.square(difference)), tf.subtract(tf.abs(difference), 0.5))

  # only compute bbox loss over positive ground truth boxes
  # processed_result = tf.where(tf.equal(class_labels, 1.0), result, tf.zeros([BATCH_SIZE, 228, 228, 6], tf.float32))
  processed_result = tf.multiply(result, class_labels)
  return tf.reduce_mean(processed_result)

def custom_cross_entropy(class_labels, class_preds):
    # ones = tf.fill([BATCH_SIZE, 228, 228, 1], 1.)
    # eps = tf.fill([BATCH_SIZE, 228, 228, 1], 0.0000001)
    lolz = tf.where(tf.equal(class_labels, 1), -tf.log(tf.add(class_preds, 0.0000001)), -tf.log(tf.add(1 - class_preds, 0.0000001)))
    return tf.reduce_mean(lolz)

if __name__ == "__main__":

    class_loss = custom_cross_entropy(class_labels=y_class, class_preds=output_class)
    box_loss = smooth_L1(box_labels=y_box, box_preds=output_box, class_labels=y_class)
    pixor_loss = class_loss + box_loss

    #A step to minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-4).minimize(pixor_loss)

    # RUN THINGS

    saver = tf.train.Saver()

    with tf.Session() as sess:

      #initialize everything
      sess.run(tf.global_variables_initializer())
      num_epochs = 50

      per_epoch_train_loss = 0
      lowest_val_loss = np.inf

      TRAIN_LEN = 301
      VAL_LEN = 38

      batch_indices = np.arange(TRAIN_LEN)
      val_batch_indices = np.arange(VAL_LEN)

      train_base_path = '../data_path/pixor/train'
      val_base_path = '../data_path/pixor/val'

      for epoch in range(num_epochs):
        per_epoch_train_loss = 0
        logging.info("\nepoch " + str(epoch))

        np.random.shuffle(batch_indices)

        # RIGHT NOW IF DOESN'T PERFECTLY DIVIDE IT DOESN'T COVER REMAINING, MIGHT WANT TO CHANGE THIS
        num_batches = TRAIN_LEN // BATCH_SIZE

        for batch_number in range(0, num_batches):
          logging.info("batch " + str(batch_number))
          start_idx = batch_number * BATCH_SIZE
          end_idx = start_idx + BATCH_SIZE

          batch_images, batch_boxes, batch_classes = get_batch(start_idx, BATCH_SIZE, batch_indices, train_base_path)

          # train on the batch
          _, b_loss, c_loss, batch_train_loss = sess.run([train_step, box_loss, class_loss, pixor_loss], feed_dict =
            {x: batch_images,
            y_box: batch_boxes,
            y_class: batch_classes})

          per_epoch_train_loss += batch_train_loss
          # print("overall batch loss: " + str(batch_train_loss))
          # print("box loss: " + str(b_loss))
          # print("class loss: " + str(c_loss))


        # at each epoch, print training and validation loss
        val_images, val_boxes, val_classes = get_batch(0, VAL_LEN, val_batch_indices, val_base_path)
        val_loss = sess.run([pixor_loss], feed_dict = {x: val_images,
          y_box: val_boxes, y_class: val_classes})
        logging.info('epoch %d, training loss %g' % (epoch, per_epoch_train_loss))
        logging.info('epoch %d, validation loss %g' % (epoch, val_loss[0]))

        # checkpoint model if best so far


        # checkpoint model if best so far
        if val_loss[0] < lowest_val_loss:
            lowest_val_loss = val_loss
            saver.save(sess, 'ckpt/', global_step=epoch)

