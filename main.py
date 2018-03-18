# Imports
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import shuffle

# Tensorflow logging verbosity
tf.logging.set_verbosity(tf.logging.INFO)
# Default dataset directory
DATASET_DIR = "datasets/bonn/"


# Loads files from given dataset directory
def load_files(dataset_directory):
    print("Reading files...")
    onlyfiles = [f for f in listdir(dataset_directory) if isfile(join(dataset_directory, f))]

    print("Found " + str(len(onlyfiles)) + " files in directory: " + dataset_directory)

    content = []
    labels = []
    for i in range(0, len(onlyfiles)):
        with open(dataset_directory + onlyfiles[i], 'r') as content_file:
            read_content = content_file.read().splitlines()
            mapped_content = list(map(np.float32, read_content))
            list.append(content, np.array(mapped_content))
            list.append(labels, onlyfiles[i][:1])

    # Changing string values to numbers
    labels_set = set(labels)
    labels_dictionary = {}
    index = 1
    for set_element in labels_set:
        labels_dictionary[set_element] = index
        index += 1

    for i in range(0, len(labels)):
        labels[i] = int(labels_dictionary.get(labels[i]))

    # print(len(content))
    # plt.plot(content[0])
    # plt.show()
    print("Read files")
    return np.array(content), np.array(labels)


# Shuffles data and labels preserving consistency
def shuffle_data(content, labels):
    print("Shuffling data...")
    indices = list(range(0, len(content)))
    shuffle(indices)
    for i in range(0, len(content)):
        content[i], content[indices[i]] = content[indices[i]], content[i]
        labels[i], labels[indices[i]] = labels[indices[i]], labels[i]
    print("Shuffled data")
    return content, labels


# Splits data into test and training set
def get_test_and_training_data(content, labels):
    print("Splitting data into sets...")
    content_length = len(content)
    content_length /= 10.0
    content_length = int(content_length)

    # Returns test data first, then training data
    print("Split data into sets")
    return content[:content_length], labels[:content_length], content[content_length:], labels[content_length:]


# Tensorflow example
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 4097, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 65536])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    content, labels = load_files(DATASET_DIR)
    content, labels = shuffle_data(content, labels)
    eval_data, eval_labels, train_data, train_labels = get_test_and_training_data(content, labels)

    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    eeg_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/eeg_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    eeg_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
