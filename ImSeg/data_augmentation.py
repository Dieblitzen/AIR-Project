from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

datagenX = ImageDataGenerator(featurewise_center=True,
                              rotation_range=90,
                              channel_shift_range=150.0,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='constant', 
                              cval=190)
datagenY = ImageDataGenerator(rotation_range=90,
                              channel_shift_range=150.0,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='constant', 
                              cval=190)

seed = tf.random.uniform(1e6)
for xBatch, yBatch in (datagenX.flow(Xs, batch_size=1, seed=seed),
                       datagenY.flow(Ys, batch_size=1, seed=seed)):
    # stuff