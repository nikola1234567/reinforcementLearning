import keras.callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from configurations import TENSORBOARD_LOGS_DIR
from TensorBoard.utils import plot_confusion_matrix
from tensorboard.plugins.hparams import api as hp
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

    def save_hparams(self, number_of_layers, hidden_size, learning_rate, accuracy):
        HP_NUM_LAYERS = hp.HParam("number_of_layers")
        HP_HIDDEN_SIZE = hp.HParam("hidden_size")
        HP_LEARNING_RATE = hp.HParam("learning_rata")

        hparams = {
            HP_NUM_LAYERS: number_of_layers,
            HP_HIDDEN_SIZE: hidden_size,
            HP_LEARNING_RATE: learning_rate
        }

        writer_path = self.create_hparam_dir(number_of_layers=number_of_layers,
                                             hidden_size=hidden_size,
                                             learning_rate=learning_rate)
        with tf.summary.create_file_writer(writer_path).as_default():
            hp.hparams(hparams)
            tf.summary.scalar("accuracy", accuracy, step=1)

    def create_log_dir(self):
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)

    def create_inner_log_dir(self, step):
        self.create_log_dir()
        path = os.path.join(self.log_dir_path, str(step))
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def create_hparam_dir(self, number_of_layers, hidden_size, learning_rate):
        path = self.create_inner_log_dir(step="hyperParameters")
        run_dir = (
                "num_layers_"
                + str(number_of_layers)
                + "hidden_size_"
                + str(hidden_size)
                + "learning_rate_"
                + str(learning_rate)
        )
        dir_path = os.path.join(path, run_dir)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return dir_path


class TensorBoardStandardManager(TensorBoardCustomManager):

    def callback(self, iteration):
        path = self.create_inner_log_dir(step=iteration)
        return keras.callbacks.TensorBoard(log_dir=path,
                                           profile_batch='500,520')
