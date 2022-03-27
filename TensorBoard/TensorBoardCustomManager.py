import keras.callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from configurations import TENSORBOARD_LOGS_DIR
from TensorBoard.utils import plot_confusion_matrix
import os

LOSS = "Loss"
ACCURACY = "Accuracy"
MSE = "Mean square error"


class TensorBoardCustomManager:

    def __init__(self, name):
        self.name = name + str('\\')
        self.log_dir_path = os.path.join(TENSORBOARD_LOGS_DIR, name)

    def save(self, loss, accuracy, mse, step):
        self.create_log_dir()
        writer = tf.summary.create_file_writer(self.log_dir_path)
        with writer.as_default():
            tf.summary.scalar(LOSS, loss, step=step)
            tf.summary.scalar(ACCURACY, accuracy, step=step)
            tf.summary.scalar(MSE, mse, step=step)
        writer.flush()

    def save_confusion_matrix(self, step, confusion, class_names):
        path = self.create_inner_log_dir(step="confusionMatrix")
        writer = tf.summary.create_file_writer(logdir=path)
        with writer.as_default():
            tf.summary.image(
                "Confusion Matrix",
                plot_confusion_matrix(confusion, class_names),
                step=step,
            )

    def create_log_dir(self):
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)

    def create_inner_log_dir(self, step):
        self.create_log_dir()
        path = os.path.join(self.log_dir_path, str(step))
        if not os.path.exists(path):
            os.mkdir(path)
        return path


class TensorBoardStandardManager(TensorBoardCustomManager):

    def callback(self, iteration):
        path = self.create_inner_log_dir(step=iteration)
        return keras.callbacks.TensorBoard(log_dir=path)
