import numpy as np
from PIL import Image
import os
import os.path as osp
import imageio

TILE_SIZE = 224
BASE_PATH = '../data_path/pixor/train'


def standardize_stats(is_img=False):

    if is_img:
        path = osp.join(BASE_PATH, 'images')
        train_len = len(os.listdir(path))
        combined_data = np.zeros((train_len, TILE_SIZE, TILE_SIZE, 3))
    else: #calculate stats for boxes
        path = osp.join(BASE_PATH, 'box_annotations')
        train_len = len(os.listdir(path))
        combined_data = np.zeros((train_len, TILE_SIZE, TILE_SIZE, 6))
        
    for i in range(train_len):
        if is_img:
            img = imageio.imread(osp.join(path, f'{i}.jpg'))
            combined_data[i] = img

        else:
            annotation = np.load(osp.join(path, f'{i}.npy'))
            combined_data[i] = annotation

    print('combined_data.shape', combined_data.shape)

    reshaped_data = combined_data
   
    if not is_img:
        reshaped_data = combined_data.reshape((train_len*TILE_SIZE*TILE_SIZE), -1)
        reshaped_data = reshaped_data[np.squeeze(np.asarray(reshaped_data[:, 0])) != TILE_SIZE, :]


    np.set_printoptions(threshold=100)
    mean = reshaped_data.mean(axis = 0)
    std = reshaped_data.std(axis = 0)
    return mean, std


if __name__ == "__main__":
    mean, std = standardize_stats(True)
    print("mean: " + str(mean))
    print(mean.shape)
    print("std: " + str(std))
    print(std.shape)
    np.save("mean", mean)
    np.save("std", std)
    
    