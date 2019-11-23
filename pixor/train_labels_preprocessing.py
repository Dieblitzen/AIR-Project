import numpy as np
from PIL import Image
import os
import os.path as osp
import imageio

TILE_SIZE = 224

def standardize_stats(data_path, is_img=False):

    if is_img:
        path = osp.join(data_path, 'images')
        train_len = len(os.listdir(path))
        combined_data = np.zeros((train_len, TILE_SIZE, TILE_SIZE, 3))
    else: #calculate stats for boxes
        path = osp.join(data_path, 'box_annotations')
        train_len = len(os.listdir(path))
        combined_data = np.zeros((train_len, TILE_SIZE, TILE_SIZE, 6))
        
    for i in range(train_len):
        if is_img:
            img = imageio.imread(osp.join(path, f'{i}.jpg'))
            combined_data[i] = img

        else:
            annotation = np.load(osp.join(path, f'{i}.npy'))
            combined_data[i] = annotation

    reshaped_data = combined_data
   
    # if not is_img:
    #     # reshaped_data = combined_data.reshape((train_len*TILE_SIZE*TILE_SIZE), -1)
    #     # reshaped_data = reshaped_data[np.squeeze(np.asarray(reshaped_data[:, 0])) != TILE_SIZE, :]
    #     # print('reshape_data.shape', reshaped_data.shape)
    #     reshaped_data = reshaped_data.reshape(-1, TILE_SIZE, TILE_SIZE, 6)
    # else:
    #     reshaped_data = reshaped_data.reshape(-1, TILE_SIZE, TILE_SIZE, 3)


    np.set_printoptions(threshold=100)
    mean = reshaped_data.mean(axis = 0)
    std = reshaped_data.std(axis = 0)
    return mean, std


def passed_arguments():
  parser.add_argument('--data_path',\
                      type=str,
                      required=True,
                      help='Name of folder that contains the pixor dataset folder.')
  return args


if __name__ == "__main__":
    # args = passed_arguments()
    #dp = f'{data_path}'
    dp = 'data_path'
    bp = osp.join('..', dp, 'pixor', 'train')

    # mean.npy and std.npy are stats about the images 
    mean, std = standardize_stats(bp, True)
    print("mean: " + str(mean[0][0][:5]))
    print(mean.shape)
    print("std: " + str(std[0][0][:5]))
    print(std.shape)
    np.save("mean", mean)
    np.save("std", std)

    # train_mean.npy and train_std.npy are stats about the bounding boxes
    train_mean, train_std = standardize_stats(bp, False)
    print("train_mean: " + str(train_mean[0][0][:5]))
    print(train_mean.shape)
    print("train_std: " + str(train_std[0][0][:5]))
    print(train_std.shape)
    np.save("train_mean", train_mean)
    np.save("train_std", train_std)
    
    