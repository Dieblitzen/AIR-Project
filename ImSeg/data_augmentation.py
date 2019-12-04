from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

DATA_GEN_X =ImageDataGenerator(rotation_range=90,
                               horizontal_flip=True,
                               vertical_flip=True,
                               channel_shift_range=64,
                               fill_mode='constant',
                               cval=0)
DATA_GEN_Y = ImageDataGenerator(rotation_range=90,
                               horizontal_flip=True,
                               vertical_flip=True,
                               channel_shift_range=1e-10,
                               fill_mode='constant', 
                               cval=0)

def augment_data(images, annotations, multiplier=1, seed=0):
  """
  Augments images and label masks
  Requires:
    images: n x IMAGE_SIZE x IMAGE_SIZE x 3 numpy arrays of images
    annotations: n x IMAGE_SIZE x IMAGE_SIZE x c numpy array of masks
      where c is the number of classes 
  Optional:
  -multiplier (default 1): number of times each image should be augmented
  -seed (default 0): seed for random image augmentation
  Returns:
    Returns a batch of the origial images (n images)
       and augmented images (multiplier * n images)
    aug_images: ((multiplier + 1) * n) x IMAGE_SIZE x IMAGE_SIZE x 3
      of numpy arrays of augmented images
    aug_annotations: ((multiplier + 1) * n) x IMAGE_SIZE x IMAGE_SIZE x c
      numpy array of masks where c is the number of classes
  """
  batch_size = images.shape[0]
  imageGen = DATA_GEN_X.flow(images, batch_size=batch_size, seed=seed)
  labelGens = []
  
  for i in range(annotations.shape[3]):
    labelGens.append(DATA_GEN_Y.flow(annotations[:,:,:,i:i + 1], batch_size=batch_size, seed=seed))

  for i in range(multiplier):
    aug_images = imageGen.next()

    # (c length list of n x IMAGE_SIZE x IMAGE_SIZE x 1) -> list of n x IMAGE_SIZE x IMAGE_SIZE x c
    aug_labels = [] 
    for classLabelGen in labelGens:
      aug_labels.append(classLabelGen.next())

    aug_labels = np.array(aug_labels) # c x n x Img x Img x 1
    aug_labels = np.moveaxis(aug_labels[:,:,:,:, 0], 0, -1)
    
    images = np.concatenate((images, aug_images)).astype(np.uint8)
    annotations = np.concatenate((annotations, aug_labels)).astype(np.uint8)

  return images, annotations