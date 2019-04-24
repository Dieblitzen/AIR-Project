import numpy as np
from PIL import Image

def standardize_stats(TRAIN_LEN, base_path):
    combined_data = np.zeros((TRAIN_LEN, 228, 228, 6))
    for i in range(TRAIN_LEN):
        annotation = np.load(base_path + '/box_annotations/' + str(i) + '.npy')
        combined_data[i] = annotation
    reshaped_data = combined_data.reshape((TRAIN_LEN*228*228), 6)
    reshaped_data = reshaped_data[np.squeeze(np.asarray(reshaped_data[:, 1])) != 228., :]
    np.set_printoptions(threshold=100)
    print(reshaped_data)
    mean = reshaped_data.mean(axis = 0)
    std = reshaped_data.std(axis = 0)
    return mean, std

if __name__ == "__main__":
    TRAIN_LEN = 301
    base_path = '../WhitePlains_data/pixor/train'
    mean, std = standardize_stats(TRAIN_LEN, base_path)
    print("mean: " + str(mean))
    print(mean.shape)
    print("std: " + str(std))
    print(std.shape)
    np.save("train_mean", mean)
    np.save("train_std", std)
    
    