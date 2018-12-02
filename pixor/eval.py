import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from data_extract import extract_data

sess = tf.InteractiveSession()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))


# Access saved Variables directly
print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()

val_data = graph.get_tensor_by_name("val_data:0")
val_classlabels = graph.get_tensor_by_name("val_classlabels:0")
val_boxlabels = graph.get_tensor_by_name("val_boxlabels:0")

feed_dict = {x: val_data, 
        y_box: val_boxlabels, y_class: val_classlabels}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("pixor_loss:0")
print('validation loss %g' % (sess.run(op_to_restore,feed_dict)))

preds = output_box.eval(feed_dict)
suppressed_preds = None
mean_IOU = sess.run(tf.metrics.mean_iou(suppressed_preds, val_boxlabels, 2))
print('mean IOU: %g' % (mean_IOU))