'''Sample Cross Entropy implementation with Tensorflow'''
import tensorflow as tf


softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

my_feed_dict = {softmax: softmax_data, one_hot: one_hot_data}
with tf.Session() as sess:
    output = sess.run(cross_entropy, feed_dict=my_feed_dict)
print(output)
