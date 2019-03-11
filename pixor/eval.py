import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from network import get_batch

VAL_LEN = 38
val_batch_indices = np.arange(VAL_LEN)
val_base_path = '../data_path/pixor/val'

sess = tf.InteractiveSession()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('ckpt/-40.meta')
saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))

# Access saved Variables directly
# print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()

# for op in graph.get_operations():
    # print(op.name)

val_images, val_boxes, val_classes = get_batch(0, VAL_LEN, val_batch_indices, val_base_path)

x = graph.get_tensor_by_name("Placeholder:0")
y_box = graph.get_tensor_by_name("Placeholder_1:0")
y_class = graph.get_tensor_by_name("Placeholder_2:0")

feed_dict = {x: val_images, y_box: val_boxes, y_class: val_classes}

#Now, access the op that you want to run. 
# op_to_restore = graph.get_tensor_by_name("pixor_loss:0")
# print('validation loss %g' % (sess.run(op_to_restore,feed_dict)))

output_box = graph.get_tensor_by_name("output_box")
output_class = graph.get_tensor_by_name("output_class")

# preds = output_box.eval(feed_dict)
box_preds, class_preds = sess.run([output_box, output_class], feed_dict)

print("box preds shape: ")
print(tf.shape(box_preds))
for idx in range(0, tf.shape(box_preds)[0]):
    np.save('../data_path/box_predictions/ + idx', box_preds[idx])
    np.save('../data_path/class_predictions/ + idx', class_preds[idx])
    
# suppressed_preds = None
# mean_IOU = sess.run(tf.metrics.mean_iou(suppressed_preds, val_boxlabels, 2))
# print('mean IOU: %g' % (mean_IOU))