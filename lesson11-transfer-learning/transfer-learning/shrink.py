import pickle
import tensorflow as tf
from collections import Counter
from sklearn.utils import shuffle
import os
from shutil import copyfile

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('dataset', '', "Name of the output file with reduced number of examples.")
flags.DEFINE_integer('size', 100, 'Number of examples per class to keep')


def main(_):
    # load bottleneck data
    training_file = os.path.join("data", "_".join([FLAGS.network, FLAGS.dataset, "100_bottleneck_features_train.p"]))

    print("Opening %s"%(training_file))
    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    X_train = train_data['features']
    y_train = train_data['labels']

    print(X_train.shape, y_train.shape)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    keep_indices = []
    keep_counter = Counter()

    for i, label in enumerate(y_train.reshape(-1)):
        if keep_counter[label] < FLAGS.size:
            keep_counter[label] += 1
            keep_indices.append(i)

    X_train_small = X_train[keep_indices]
    y_train_small = y_train[keep_indices]

    print(X_train_small.shape, y_train_small.shape)

    data = {'features': X_train_small, 'labels': y_train_small}

    # Backup the original
    print("Backing up %s to %s.bak"%(training_file, training_file))
    copyfile(training_file, training_file + ".bak")

    print("Writing to {}".format(training_file))
    pickle.dump(data, open(training_file, 'wb'))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
