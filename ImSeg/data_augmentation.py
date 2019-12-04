from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

def augment_data(images, annotations, rotate_range=0, flip=False, 
                  channel_shift_range=1e-10, multiplier=0, seed=0):
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
  data_gen_X =ImageDataGenerator(rotation_range=rotate_range,
                                horizontal_flip=flip,
                                vertical_flip=flip,
                                channel_shift_range=channel_shift_range,
                                fill_mode='constant',
                                cval=0)
  data_gen_Y = ImageDataGenerator(rotation_range=rotate_range,
                                horizontal_flip=flip,
                                vertical_flip=flip,
                                channel_shift_range=1e-10,
                                fill_mode='constant', 
                                cval=0)


  batch_size = images.shape[0]
  imageGen = data_gen_X.flow(images, batch_size=batch_size, seed=seed)
  labelGens = []
  
  for i in range(annotations.shape[3]):
    labelGens.append(data_gen_Y.flow(annotations[:,:,:,i:i + 1], batch_size=batch_size, seed=seed))

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