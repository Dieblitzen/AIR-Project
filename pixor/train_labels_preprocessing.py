import numpy as np
from PIL import Image
import os
import os.path as osp

TILE_SIZE = 224
BASE_PATH = '../data_path/pixor/train'
IMG_PATH = osp.join(BASE_PATH, 'images')
BOX_ANNOT_PATH = osp.join(BASE_PATH, 'box_annotations')
TRAIN_LEN = len(os.listdir(BOX_ANNOT_PATH))

def standardize_box_stats(base_path):
    combined_data = np.zeros((TRAIN_LEN, TILE_SIZE, TILE_SIZE, 6))
    
    for i in range(TRAIN_LEN):
        annotation = np.load(osp.join(BOX_ANNOT_PATH, str(i) + '.npy'))
        combined_data[i] = annotation

    reshaped_data = combined_data.reshape((TRAIN_LEN*TILE_SIZE*TILE_SIZE), 6)
    reshaped_data = reshaped_data[np.squeeze(np.asarray(reshaped_data[:, 1])) != TILE_SIZE, :]
    np.set_printoptions(threshold=100)
    mean = reshaped_data.mean(axis = 0)
    std = reshaped_data.std(axis = 0)
    return mean, std

def standardize_img_stats(base_path):
    combined_data = np.zeros((TRAIN_LEN, TILE_SIZE, TILE_SIZE, 6))
    
    for i in range(TRAIN_LEN):
        annotation = np.load(osp.join(BOX_ANNOT_PATH, str(i) + '.npy'))
        combined_data[i] = annotation

    reshaped_data = combined_data.reshape((TRAIN_LEN*TILE_SIZE*TILE_SIZE), 6)
    reshaped_data = reshaped_data[np.squeeze(np.asarray(reshaped_data[:, 1])) != TILE_SIZE, :]
    np.set_printoptions(threshold=100)
    mean = reshaped_data.mean(axis = 0)
    std = reshaped_data.std(axis = 0)
    return mean, std

if __name__ == "__main__":
    mean, std = standardize_box_stats(BASE_PATH)
    print("mean: " + str(mean))
    print(mean.shape)
    print("std: " + str(std))
    print(std.shape)
    np.save("train_mean", mean)
    np.save("train_std", std)
    
    