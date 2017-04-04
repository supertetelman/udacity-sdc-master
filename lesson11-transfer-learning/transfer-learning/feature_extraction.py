import pickle
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from traffic_sign_classifier import TrafficClassifier


def download_data(training_file, validation_file):
    '''Download the cifar10 dataset and parse it to the specified files'''
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
    # it's a good idea to flatten the array.
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    train = { 'features': X_train, 'labels': y_train }
    validation = { 'features': X_test, 'labels': y_test }
    with open(training_file, 'wb') as f:
        train_data = pickle.dump(train, f)
    with open(validation_file, 'wb') as f:
        validation_data = pickle.dump(validation, f)


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def train_old_model(X, y, X_cv, y_cv, nb_classes, epochs):
    '''Create an instance of the TrafficClassifier model and train it for new y/x'''
    tc = TrafficClassifier(nb_classes, True)
    with tf.Session() as sess:
        tc.train(tc.preprocess(X), y, tc.preprocess(X_cv), y_cv, 
                restore=False, model='models/cifar_lenet', EPOCHS=epochs)
    print("Done training old model")


def main(_):
    if FLAGS.download:
        download_data(FLAGS.training_file, FLAGS.validation_file)

    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # Use existing model as a baseline
    if FLAGS.run_old:
        train_old_model(X_train, y_train, X_val, y_val, nb_classes, FLAGS.epochs)

    # define model
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'validation.p', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'train.p', "Bottleneck features validation file (.p)")
flags.DEFINE_bool('download', False, "Download the CIFAR10 data before doing anything else.")
flags.DEFINE_bool('run_old', False, "Train and run the old model.")
flags.DEFINE_integer('epochs', 1, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
