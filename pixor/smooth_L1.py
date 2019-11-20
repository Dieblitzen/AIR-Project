import tensorflow as tf
import numpy as np

TILE_SIZE = 224
""" Implements smooth L1 on each dimension. Erases loss for negative pixel locations. """
def smooth_L1(box_labels, box_preds, class_labels):
	difference = tf.subtract(box_preds, box_labels)
	result = tf.where(tf.abs(difference) < 1, tf.multiply(0.5, tf.square(difference)), tf.abs(difference) - 0.5)
	class_labels = tf.cast(tf.clip_by_value(class_labels, clip_value_min=0, clip_value_max=1), dtype=tf.float32)
	# only compute bbox loss over positive ground truth boxes
	processed_result = tf.multiply(result, class_labels)

	return tf.reduce_mean(processed_result)

def decode_smooth_L1(box_labels, box_preds, class_labels):
	difference = tf.subtract(box_preds, box_labels)
	result = tf.where(tf.abs(difference) < 1, tf.multiply(0.5, tf.square(difference)), tf.abs(difference) - 0.5)
	
	# only compute bbox loss over positive ground truth boxes
	reshaped_result = tf.reshape(result, [-1, TILE_SIZE, TILE_SIZE, 8])
	class_labels = tf.cast(tf.clip_by_value(class_labels, clip_value_min=0, clip_value_max=1), dtype=tf.float32)


	processed_result = tf.multiply(reshaped_result, class_labels)
	reshaped_processed = tf.reshape(processed_result, [-1, TILE_SIZE, TILE_SIZE, 4, 2])

	return tf.reduce_mean(reshaped_processed)


if __name__ == "__main__":

	# BELOW IS A TEST CASE. ANSWER SHOULD BE ~0.58167

	sess = tf.InteractiveSession()

	box_preds = [[1, 0.5, 0.3]]
	box_labels = [[0, 0.2, 2]]
	class_labels = tf.convert_to_tensor([[1.0]])
	box_preds = tf.convert_to_tensor(box_preds)
	box_labels = tf.convert_to_tensor(box_labels)
	result = smooth_L1(box_labels, box_preds, class_labels)
	print("result is: " + str(result.eval())) 
