'''Simple Hello World Tensorflow script'''
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

with tf.Session() as sess:
    # Run the tf.constant operations in the session
    for tensor in [ hello_constant, A, B, C ]:
        output = sess.run(tensor)
        print(output)
        