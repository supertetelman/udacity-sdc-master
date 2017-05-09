import pickle
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle


# Load traffic signs data.
with open('./train.p', 'rb') as f:
    data = pickle.load(f)

# Split data into training and validation sets.
SPLIT = 0.3
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=SPLIT)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
resized = tf.image.resize_images(x, [227, 227])


# Freeze AlexNet
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)

# Determine output shape of fc7, classes
nb_classes = 43
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# Add the final layer for traffic sign classification.
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# # Define loss, training, accuracy operations.

# Define learning rate
RATE = 0.001

# Define the one hot encoded y
y = tf.placeholder(tf.int64, (None))
one_hot_y = tf.one_hot(y, nb_classes)

# Define Error function, I am using cross entropy
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

# Define the loss function, this will minimize overall Error
loss_operation = tf.reduce_mean(cross_entropy)

# Define Optimizer & Learning Rate
optimizer =  tf.train.AdamOptimizer(learning_rate = RATE)

# Define the training tensor. This will just minimize our loss function
train = optimizer.minimize(loss_operation)

# Define tensor to determine correct predictions and calculate percentage correct
correct_prediction = tf.cast(tf.equal(tf.arg_max(logits, 1), y), tf.float32)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train and evaluate the feature extraction model.
EPOCHS = 10
BATCH_SIZE = 256

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,EPOCHS):
        total_loss = 0
        total_acc = 0

        print("Starting new epoch")
        X_train, y_train = shuffle(X_train, y_train)


        t0 = time.time()
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            myfeed = {x: X_train[offset:end], y: y_train[offset:end]}
            sess.run(train, feed_dict=myfeed)

        sess = tf.get_default_session()
        for offset in range(0, X_test.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            myfeed = {x: X_test[offset:end], y: y_test[offset:end]}

            loss, acc = sess.run([loss_operation, accuracy_operation], feed_dict=myfeed)

            total_loss += (loss * X_test[offset:end].shape[0])
            total_acc += (acc * X_test[offset:end].shape[0])
        val_loss = total_loss/X_test.shape[0]
        val_acc = total_acc/X_test.shape[0]

        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
