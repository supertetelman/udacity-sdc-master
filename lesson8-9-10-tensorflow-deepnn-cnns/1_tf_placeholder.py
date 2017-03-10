'''Simple tv.placeholder() introductory script'''
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.string)
    z = tf.placeholder(tf.float32)

    my_feed_dict = {x: 123, y: "Input String", z: 123.456}

    with tf.Session() as sess:
        print(my_feed_dict)
        # Pass data through a single tensor
        output = sess.run(x, feed_dict=my_feed_dict)
        print(output)
        output = sess.run(y, feed_dict=my_feed_dict)
        print(output)
        output = sess.run(z, feed_dict=my_feed_dict)
        print(output)

        # Pass data through multiple tensors
        output = sess.run([x, y, z], feed_dict=my_feed_dict)
        print(output)


run()
