import tensorflow as tf
from tensorflow.python.ops import variables


x = tf.placeholder(tf.float32, (None, 256, 256, 3), name='input')
y_ = tf.placeholder(tf.int32, (None), name='label')

x = tf.layers.conv2d(x, 64, (3, 3), padding='same', name='cv1')
x = tf.layers.batch_normalization(x, training=True, name='nb1')
x = tf.nn.relu(x)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

print(tf.global_variables())

print(tf.trainable_variables())

print(tf.moving_average_variables())

merge = tf.summary.merge_all()

print("hello")