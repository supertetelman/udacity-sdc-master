'''Sample implementation of Dropout networks with Tensorflow'''
import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

keep_prob = tf.placeholder(tf.float32)

f1 = tf.add(tf.matmul(features, weights[0]), biases[0])
a1 = tf.nn.relu(f1)
f2 = tf.nn.dropout(a1, keep_prob)
a2 = f2
logits = tf.add(tf.matmul(a2, weights[1]), biases[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits, feed_dict={keep_prob: 0.5})
    print(output)