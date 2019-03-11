import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from network import get_batch

VAL_LEN = 38
val_batch_indices = np.arange(VAL_LEN)
val_base_path = '../data_path/pixor/val'

# sess = tf.InteractiveSession()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('ckpt/-40.meta')

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    
    saver.restore(sess,tf.train.latest_checkpoint('ckpt/'))
    
    sess.run(init_op)

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()

#     for op in graph.get_operations():
#         print(op.name)

    val_images, val_boxes, val_classes = get_batch(0, VAL_LEN, val_batch_indices, val_base_path)

    x = graph.get_tensor_by_name("x:0")
    y_box = graph.get_tensor_by_name("y_box:0")
    y_class = graph.get_tensor_by_name("y_class:0")

    output_box = graph.get_tensor_by_name("output_box/BiasAdd:0")
    output_class = graph.get_tensor_by_name("output_class/Sigmoid:0")

    box_preds, class_preds = sess.run([output_box, output_class], feed_dict={x: val_images, y_box: val_boxes, y_class: val_classes})

    print(box_preds.shape)
    print(class_preds.shape)
    for idx in range(0, box_preds.shape[0]):
        np.save('../data_path/pixor/val/box_predictions/' + str(idx), box_preds[idx])
        np.save('../data_path/pixor/val/class_predictions/' + str(idx), class_preds[idx])

    # suppressed_preds = None
    # mean_IOU = sess.run(tf.metrics.mean_iou(suppressed_preds, val_boxlabels, 2))
    # print('mean IOU: %g' % (mean_IOU))
