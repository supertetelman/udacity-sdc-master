'''A simple script to show case simple math tensors'''
import tensorflow as tf

x = tf.placeholder(tf.int32) # x = 10
y = tf.placeholder(tf.int32) # y = 2
z1 = tf.subtract(tf.cast(tf.divide(x,y), tf.int32), 1) # z = x/y - 1 as int
z2 = tf.subtract(tf.divide(x,y), 1.0) # z = x/y - 1 as float
z3 = tf.add(x,y) # z = x + y
z4 = tf.multiply(x,y) # z = x * y
z5 = tf.subtract(x,y) # z = x - y

my_feed_dict = {x: 10, y: 2}

with tf.Session() as sess:
    for z in [z1, z2, z3, z4, z5]:
        output = sess.run(z, feed_dict=my_feed_dict)
        print(output)
