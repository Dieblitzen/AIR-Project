import argparse
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

import pixor_model
from pixor_model import PixorModel


parser = argparse.ArgumentParser(description='Arguments for GAN with rotation loss')
parser.add_argument('--gpu', type=int, default=0, help='0 - run with CPU; 1 - run with GPU')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--logfile_name', type=str, default='PIXOR_logfile', help='File to save logging info')
parser.add_argument('--tile_size', type=int, default=224, help='Size of tile images')
parser.add_argument('--data_path', type=str, default='data_path', help='Name of data folder')
parser.add_argument('--num_classes', type=int, default=6, help='number of building classes')

flags = parser.parse_args()
##### SETTINGS #####

NUM_EPOCHS = flags.num_epochs
BATCH_SIZE = flags.batch_size
TILE_SIZE = flags.tile_size
IMAGE_SIZE = (TILE_SIZE, TILE_SIZE, 3)
LOGFILE_NAME = flags.logfile_name
GPU = flags.gpu
NUM_CLASSES = flags.num_classes
DATA_FILE_NAME = flags.data_path
TRAIN_BASE_PATH = os.path.join('..', DATA_FILE_NAME, 'pixor', 'train')
TRAIN_LEN = len(os.listdir(os.path.join(TRAIN_BASE_PATH, 'images')))

VAL_BASE_PATH = os.path.join('..', DATA_FILE_NAME, 'pixor', 'val')
VAL_LEN = len(os.listdir(osp.join(VAL_BASE_PATH, 'images')))
##### End of SETTINGS #####

logging.basicConfig(level=logging.INFO, filename=LOGFILE_NAME,
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
sys.path.append("..")


# launch session to connect to C++ computation power
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options)) if GPU else tf.Session()


def pixor_to_corners_tf(box):
    center_x, center_y, cos_angle, sin_angle, width, length = box
    four_corners = [(center_x+width//2, center_y+length//2),
        (center_x+width//2, center_y-length//2),
        (center_x-width//2, center_y-length//2),
        (center_x-width//2, center_y+length//2)]

    rotated_corners = [rotate_point(corner, center_x, center_y, cos_angle, sin_angle) for corner in four_corners]
    return rotated_corners


def find_angle(box):
    try:
        angle = np.arccos(box[2])
    except:
        try:
            angle = np.arcsin(box[3])
        except:
            angle = np.arccos(round(box[2]%math.pi, 4))
    return angle

        
def viz_preds(box_preds, class_preds):
    vis_val_images, vis_val_boxes, vis_val_classes = pixor_model.get_batch(0, flags, path=VAL_BASE_PATH,  norm=False)


    true_pos = 0.
    false_pos = 0.
    
    for i in range(len(vis_val_images)):
        
        logging.info("image " + str(i))
        unique_boxes_set = set()
        boxes_in_image = []
        nms_boxes_in_image = []
        box_classes_in_image = []
        boxes_reduced = []
        image = vis_val_images[i].astype(int)
        
        max_op = np.maximum(class_preds[i] - 0.5, np.zeros((class_preds[i].shape)))
        pos_indices = np.nonzero(max_op)
        pos_indices = pos_indices[:-1]
        
        # only contains the positive box predictions for image i
        pos_box_preds = box_preds[i][pos_indices]
        
        for j in range(0, pos_box_preds.shape[0]):
            r = pos_indices[0][j]
            c = pos_indices[1][j]
            
            curr_norm_box_pred = pos_box_preds[j]
            curr_box_pred = (curr_norm_box_pred*train_std) + train_mean

            center_x = (c) + (int(curr_box_pred[0]))
            center_y = (r) + (int(curr_box_pred[1]))
            center = np.array([center_x, center_y])
            box = np.concatenate([center, curr_box_pred[2:]])
            angle = find_angle(box)

            nms_box = ((box[0], box[1]), (box[-2], box[-1]), angle)
            if tuple(curr_box_pred[2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(curr_box_pred[2:]))                               
                if not np.isnan(cv2.boxPoints(nms_box)).any():
                    nms_boxes_in_image.append(nms_box)
                    box_classes_in_image.append(class_preds[i][r,c][0])
                boxes_in_image.append(box)
  
        selected_indices = nms.rboxes(nms_boxes_in_image, box_classes_in_image)
        boxes_reduced = [boxes_in_image[i] for i in selected_indices]
        unique_val_boxes = meanAP.extract_unique_labels(vis_val_boxes[i])
        
        visualize_data.visualize_bounding_boxes(image, boxes_reduced, True, i, 'output_visualized', 'blue')
        visualize_data.visualize_bounding_boxes(image, unique_val_boxes, True, i, 'label_visualized', 'green')
        
    
def get_MAP(box_preds, class_preds):
    vis_val_images, vis_val_boxes, vis_val_classes = pixor_model.get_batch(0, flags, path=val_base_path, norm=False)
    true_pos = 0.
    false_pos = 0.
    
    for i in range(len(vis_val_images)):
        logging.info("image " + str(i))
        unique_boxes_set = set()
        boxes_in_image = []
        nms_boxes_in_image = []
        box_classes_in_image = []
        boxes_reduced = []
        image = vis_val_images[i].astype(int)
        
        max_op = np.maximum(class_preds[i] - 0.5, np.zeros((class_preds[i].shape)))
        pos_indices = np.nonzero(max_op)
        pos_indices = pos_indices[:-1]
        
        # only contains the positive box predictions for image i
        pos_box_preds = box_preds[i][pos_indices]
        
        for j in range(0, pos_box_preds.shape[0]):
            r = pos_indices[0][j]
            c = pos_indices[1][j]
            
            
            curr_norm_box_pred = pos_box_preds[j]
            curr_box_pred = (curr_norm_box_pred*train_std) + train_mean

            center_x = (c) + (int(curr_box_pred[0]))
            center_y = (r) + (int(curr_box_pred[1]))
            center = np.array([center_x, center_y])
            box = np.concatenate([center, curr_box_pred[2:]])
            angle = find_angle(box)

            nms_box = ((box[0], box[1]), (box[-2], box[-1]), angle)
            if tuple(curr_box_pred[2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(curr_box_pred[2:]))                               
                if not np.isnan(cv2.boxPoints(nms_box)).any():
                    nms_boxes_in_image.append(nms_box)
                    box_classes_in_image.append(class_preds[i][r,c][0])
                boxes_in_image.append(box)
        
        selected_indices = nms.rboxes(nms_boxes_in_image, box_classes_in_image, nms_threshold=0.1)
        boxes_reduced = [(boxes_in_image[i], box_classes_in_image[i]) for i in selected_indices]
        sorted_by_conf = sorted(boxes_reduced, key= lambda pair: pair[1], reverse = True)
        boxes_reduced, _ = zip(*sorted_by_conf)
        unique_val_boxes = meanAP.extract_unique_labels(vis_val_boxes[i])
        logging.info("validation boxes: ")
        logging.info(unique_val_boxes)
        val_boxes_reformatted = [((box[0], box[1]), (box[-2], box[-1]), find_angle(box)) for box in unique_val_boxes]
        boxes_reduced_reformatted = [((box[0], box[1]), (box[-2], box[-1]), find_angle(box)) for box in boxes_reduced]
        if boxes_reduced_reformatted != []:
            im_true_pos, im_false_pos = meanAP.image_meanAP(boxes_reduced_reformatted, val_boxes_reformatted[1:], .5)
            true_pos += im_true_pos
            false_pos += im_false_pos

    logging.info("mAP:")
    if true_pos + false_pos != 0:
        logging.info(true_pos/(true_pos+false_pos))
    else:
        logging.info("no predictions, mAP undefined")

def log_print(log_string):
    logging.info(log_string)
    print(log_string)

if __name__ == "__main__":
    
    # pos_weight = 60000000/2763487
    # neg_weight = 60000000/12831713
    
    #A step to minimize our cost function
    model = PixorModel(flags)
    box_loss, pixor_loss, decode_loss, decode_pixor_loss = model.get_loss()
    
    # RUN THINGS
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
      #initialize everything
        sess.run(tf.global_variables_initializer())

        per_epoch_train_loss = 0
        lowest_val_loss = np.inf

        mAP = 0.
        #opt = tf.keras.optimizers.Adam(0.1)
        #ckpt = tf.train.Checkpoint(step=tf.Variable(1))
        #manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep = 3)


        #ckpt.restore(manager.latest_checkpoint)
        

        for epoch in range(NUM_EPOCHS):
            box_preds, unnorm_class_preds, per_epoch_train_loss, per_epoch_box_loss, per_epoch_class_loss = model.train_one_epoch(epoch, sess, BATCH_SIZE, TRAIN_BASE_PATH)

            # at each epoch, print training and validation loss
            val_loss, box_preds, unnorm_class_preds, val_classes = model.evaluate(sess, VAL_BASE_PATH)
            
            log_print(f'epoch {epoch}')
            log_print(f'training loss {per_epoch_train_loss}')
            log_print(f'training class loss {per_epoch_class_loss}')
            log_print(f'training box loss {per_epoch_box_loss}')
            log_print(f'validation loss {val_loss}')
            
            # pos = np.where(class_preds >.8)
            class_preds = tf.sigmoid(unnorm_class_preds).eval()
            max_op = np.maximum(class_preds - 0.5, np.zeros((class_preds.shape)))
            pos_indices = np.nonzero(max_op)
            pos_indices = pos_indices[:-1]
                    
            # checkpoint model if best so far
            if val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                #saver.save(sess, 'ckpt/', global_step=epoch)
            # saving weights
            #ckpt.step.assign_add(1)
            #if epoch % 10 == 0:
                #save_path = manager.save()
                #print("Saved checkpoint for step {} :{}".format(int(ckpt.step),save_path))

        print('val_classes.shape', val_classes.shape) #(1, 224, 224, 1)
        temp = val_classes.flatten().astype(int)
        print('val_classes max', int(temp.max()))
        val_classes = np.zeros((temp.size, int(temp.max())+1))
        val_classes[np.arange(temp.size), temp] = 1
        print('val_classes.shape after', val_classes.shape)
        print('class_preds.shape after', class_preds.shape) #(1, 224, 224, 6)

        ap = average_precision_score(val_classes.flatten(), np.round(class_preds.flatten()))
        precision = precision_score(val_classes.flatten(), np.round(class_preds.flatten()))
        recall = recall_score(val_classes.flatten(), np.round(class_preds.flatten()))
        
        log_print(f'ap: {ap}')
        log_print(f'precision: {precision}')
        log_print(f'recall: {recall}')            

        val_loss, box_preds, unnorm_class_preds, val_classes = model.evaluate_one(sess, VAL_BASE_PATH)
        bbox = pixor_to_corners_tf(box_preds)

        #ADD PLOTING 
        p = osp.join(VAL_BASE_PATH, 'images', '1.jpg')
        im = Image.open(p)
        im_arr = np.array(im)
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(im_arr)
        f.add_subplot(1, 2, 2)
        plt.imshow(np.squeeze(val_classes))
        #plt.show(block=True)
        draw = ImageDraw.Draw(im)
        for _,points in bbox:
            p = sorted(points)
            ps = []
            ps.append(p[0])
            ps.append(p[1])
            ps.append(p[3])
            ps.append(p[2])
            draw.polygon(tuple(ps),outline="blue")
        im.show()


    #save outputs for visualizing/calculate MAP (skipping eval.py)
        if epoch % 25 == 0 and epoch != 0 and epoch != 25:
            get_MAP(box_preds, class_preds)
        if epoch == 0:
            viz_preds(box_preds, class_preds)
            
