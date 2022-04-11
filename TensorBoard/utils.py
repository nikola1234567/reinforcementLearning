import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
from sklearn.metrics import confusion_matrix
from tensorboard.plugins import projector
import os
import shutil


# Stolen from tensorflow official guide:
# https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def get_confusion_matrix(y_labels, predictions, class_names):
    cm = confusion_matrix(
        y_labels, predictions, labels=np.arange(len(class_names)),
    )

    return cm


def plot_confusion_matrix(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    # Normalize Confusion Matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3, )

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                i, j, cm[i, j], horizontalalignment="center", color=color,
            )

    plt.tight_layout()
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")

    cm_image = plot_to_image(figure)
    return cm_image


def step_to_string(step):
    return 'step_{}'.format(step)


def episode_to_string(episode):
    return 'episode_{}'.format(episode)


def episode_step_to_confusion_matrix_step(episode, step):
    step = int('{}0{}'.format(episode, step))
    return step
