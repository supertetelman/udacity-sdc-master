'''Example of tensorflow softmax use'''
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    softmax = tf.nn.softmax(logits)

    my_feed_dict = {logits: logit_data}
    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict=my_feed_dict)
    print(output)


run()
