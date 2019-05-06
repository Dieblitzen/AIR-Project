import numpy as np

## Calculate mean of dataset.
def mean_of_data(data):
  mean = 0
  for im in data:
    mean += np.mean(im)
  return mean/float(len(data))

