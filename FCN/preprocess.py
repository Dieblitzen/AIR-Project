import numpy as np

## Calculate  mean of dataset.
def mean_of_data(data):
  mean = np.mean(data, axis=0)
  return mean

## Calculate average standard deviation
def std_of_data(data):
  std = np.std(data, axis=0)
  return std