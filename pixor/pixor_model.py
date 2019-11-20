import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import logging
import visualize_data
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
from nms import nms
import cv2
import meanAP
import os
import os.path as osp
from smooth_L1 import smooth_L1, decode_smooth_L1

TILE_SIZE = 224
NUM_CLASSES = 3
BATCH_SIZE = 32
TRAIN_BASE_PATH = '../data_path/pixor/train'
TRAIN_LEN = len(os.listdir(osp.join(TRAIN_BASE_PATH, 'images')))
print('TRAIN_LEN', TRAIN_LEN)

class PixorModel(object):

    def __init__(self): 
        #Initialize expected input for images
        self.x = tf.placeholder(tf.float32, shape=(None, TILE_SIZE, TILE_SIZE, 3), name='x')
        #Initialize holder for per-pixel bounding boxes
        self.y_box = tf.placeholder(tf.float32, shape=(None, TILE_SIZE, TILE_SIZE, 6), name='y_box')
        #Initialize holder for per-pixel labels
        self.y_class = tf.placeholder(tf.int32, shape=(None, TILE_SIZE, TILE_SIZE, 1), name='y_class')
        # two convolutional layers, 3x3, 32 filters
        conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
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

   
        upsample1 = self.conv2d_transpose(input=prep_upsampling, filter_size=3, out_channels=128, stride=2, activation=tf.nn.relu)

        # postprocessing to add skip connection after upsample 6
        # [1x1, 128 channel convolution on skip ;; then add]
        processed_skip_block3 = tf.layers.conv2d(inputs=skip_block3, filters=128, kernel_size=1, padding='same', activation=tf.nn.relu)
        skipped_upsample1 = upsample1 + processed_skip_block3

        # upsample 7, 96 filters, x2
        upsample2 = self.conv2d_transpose(input=skipped_upsample1, filter_size=3, out_channels=96, stride=2, activation=tf.nn.relu)

        # postprocessing to add skip connection after upsample 7
        # [1x1, 96 channel convolution on skip ;; then add]
        processed_skip_block2 = tf.layers.conv2d(inputs=skip_block2, filters=96, kernel_size=1, padding='same', activation=tf.nn.relu)
        skipped_upsample2 = upsample2 + processed_skip_block2

        # PLACEHOLDER UPSAMPLING
        temp_final_upsample = self.conv2d_transpose(input=skipped_upsample2, filter_size=3, out_channels=96, stride=4, activation=tf.nn.relu)

        # HEADER NETWORK
        # four convolutional layers, 3x3, 96 filters
        header1 = tf.layers.conv2d(inputs=temp_final_upsample, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
        header2 = tf.layers.conv2d(inputs=header1, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
        header3 = tf.layers.conv2d(inputs=header2, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)
        header4 = tf.layers.conv2d(inputs=header3, filters=96, kernel_size=3, padding='same', activation=tf.nn.relu)

        # one convolutional layer, 3x3, 1 filter
        self.output_class = tf.layers.conv2d(inputs=header4, filters=NUM_CLASSES, kernel_size=3, padding='same', name='output_class')
        # one convolutional layer, 3x3, 6 filters
        self.output_box = tf.layers.conv2d(inputs=header4, filters=6, kernel_size=3, padding='same', name='output_box')
        # print('self.output_box.shape', self.output_box.shape) #(?, 224, 224, 6)
        
        self.get_loss()
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.pixor_loss)
        self.decode_train_step = tf.train.AdamOptimizer(1e-4).minimize(self.decode_pixor_loss)

        self.mean = np.load('mean.npy')
        self.std = np.load('std.npy')
        self.train_mean = np.load('train_mean.npy')
        self.train_std = np.load('train_std.npy')

    def get_loss(self):
        pos_weight = 1
        neg_weight = 1
        class_loss_result = self.custom_cross_entropy(class_labels=self.y_class, unnormalized_class_preds=self.output_class, class_weights=(pos_weight, neg_weight))
        class_loss = 10 * class_loss_result
        smooth_L1_loss = 100 * smooth_L1(box_labels=self.y_box, box_preds=self.output_box, class_labels=self.y_class)
        
        self.decoded_output = visualize_data.tf_pixor_to_corners(self.output_box)
        # print('decoded_output.shape', decoded_output.shape) # (?, 224, 224, 4, 2)
        self.decoded_labels = visualize_data.tf_pixor_to_corners(self.y_box)
        self.decode_loss = 100 * decode_smooth_L1(box_labels=self.decoded_labels, box_preds=self.decoded_output, class_labels=self.y_class)
        
        self.box_loss = smooth_L1_loss
        self.pixor_loss = class_loss + self.box_loss
        self.decode_pixor_loss = class_loss + self.decode_loss

        # return self.box_loss, self.pixor_loss, decode_loss, decode_pixor_loss

    # alpha is the weight of the less frequent class
    def custom_cross_entropy(self, class_labels, unnormalized_class_preds, class_weights, alpha=0.25, gamma=2.0): 
        squeezed_y = tf.squeeze(class_labels, -1) 
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unnormalized_class_preds, labels=squeezed_y)
        classify_loss = tf.reduce_mean(loss)
        return classify_loss

    """ Standard transposed convolutional layer."""
    def conv2d_transpose(self, input, filter_size, out_channels, stride, activation="None"):
        return tf.layers.conv2d_transpose(inputs=input, filters=out_channels,
            kernel_size=filter_size, strides=stride, padding='same', activation=activation) 

    def train_one_epoch(self, epoch):
        per_epoch_train_loss = 0
        per_epoch_box_loss = 0
        per_epoch_class_loss = 0

        batch_indices = np.arange(TRAIN_LEN)
        
        # logging.info("\nepoch " + str(epoch))
        print("\nepoch " + str(epoch))

        np.random.shuffle(batch_indices)
        
        # RIGHT NOW IF DOESN'T PERFECTLY DIVIDE IT DOESN'T COVER REMAINING, MIGHT WANT TO CHANGE THIS
        num_batches = TRAIN_LEN // BATCH_SIZE
        for batch_number in range(0, num_batches):
            start_idx = batch_number * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_images, batch_boxes, batch_classes = self.get_batch(start_idx, batch_indices)

            # # train on the batch
            # if epoch <= -1: 
            #     _, b_loss, c_loss, batch_train_loss= sess.run([self.train_step, self.box_loss, self.class_loss, self.pixor_loss],\
            #     feed_dict =
            #         {self.x: batch_images,
            #         self.y_box: batch_boxes,
            #         self.y_class: batch_classes})
            # else:
            _, b_loss, c_loss, batch_train_loss, box_preds, unnorm_class_preds = \
            sess.run([self.decode_train_step, self.decode_loss, self.class_loss, self.decode_pixor_loss, self.output_box, self.output_class], 
            feed_dict =
                {self.x: batch_images,
                self.y_box: batch_boxes,
                self.y_class: batch_classes})

            per_epoch_train_loss += batch_train_loss
            per_epoch_box_loss += b_loss
            per_epoch_class_loss += c_loss
        
        return box_preds, unnorm_class_preds, per_epoch_train_loss, per_epoch_box_loss, per_epoch_class_loss

    def get_batch(self, start_index, batch_indices, norm=True):
        """
        Method 3)
        Gets batch of tiles and labels associated with data start_index.

        Returns:
        [(tile_array, list_of_buildings), ...]
        """
        batch_images = np.zeros((BATCH_SIZE, TILE_SIZE, TILE_SIZE, 3))
        batch_boxes = np.zeros((BATCH_SIZE, TILE_SIZE, TILE_SIZE, 6))
        batch_classes = np.zeros((BATCH_SIZE, TILE_SIZE, TILE_SIZE, 1))
        for i in range(start_index, start_index + BATCH_SIZE):
            batch_images[i % BATCH_SIZE], batch_boxes[i % BATCH_SIZE], batch_classes[i % BATCH_SIZE] = self.get_tile_and_label(batch_indices[i], norm)

        return batch_images, batch_boxes, batch_classes

    def get_tile_and_label(self, index, norm):
        """
        Method 2)
        Gets the tile and label associated with data index.

        Returns:
        (tile_array, dictionary_of_buildings)
        """

        # Open the jpeg image and save as numpy array
        im = Image.open(TRAIN_BASE_PATH + '/images/' + str(index) + '.jpg')
        im_arr = np.array(im)
        im_arr = (im_arr - self.mean) / self.std
        
        class_annotation = np.load(TRAIN_BASE_PATH + '/class_annotations/' + str(index) + '.npy')
        class_annotation = np.expand_dims(class_annotation, -1)
        # Open the json file and parse into dictionary of index -> buildings pairs
        box_annotation = np.load(TRAIN_BASE_PATH + '/box_annotations/' + str(index) + '.npy')

        # normalizing the positive labels if norm=True
        if norm:
            clipped = np.clip(class_annotation, 0, 1)
            box_annotation = clipped * (box_annotation - self.train_mean)/self.train_std + (1 - clipped) * box_annotation
        return im_arr, box_annotation, class_annotation
