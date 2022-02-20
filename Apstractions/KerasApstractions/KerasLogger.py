from datetime import datetime
import os
import tensorflow as tf
from configurations import POLICY_WEIGHTS_DIR
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import CAR_DATASET_PATH
from Apstractions.FileApstractions.FileWorker import FileWorker


class PolicyWeightsNotFound(Exception):
    pass


class KerasLogger:

    @classmethod
    def logging_name(cls, dataset_path):
        dataset_name = FileWorker.file_name(dataset_path)
        current_datetime = datetime.now()
        cd_string = current_datetime.strftime("%d_%m_%Y_%H_%M")
        return '{}_{}_dataset.h5'.format(cd_string, dataset_name)

    @classmethod
    def logging_path(cls, dataset_path):
        logging_name = KerasLogger.logging_name(dataset_path)
        return os.path.join(POLICY_WEIGHTS_DIR, logging_name)

    @classmethod
    def save_network(cls, network_model, dataset_path):
        KerasLogger.clean_policy_log_history()
        location = KerasLogger.logging_path(dataset_path)
        network_model.save(location)

    @classmethod
    def clean_policy_log_history(cls):
        FileWorker.clean_directory(POLICY_WEIGHTS_DIR)

    @classmethod
    def load_latest_policy(cls):
        content_policy_dir = FileWorker.content_of_directory(POLICY_WEIGHTS_DIR)
        if len(content_policy_dir) == 0:
            raise PolicyWeightsNotFound()
        model_name = content_policy_dir[0]
        model_path = os.path.join(POLICY_WEIGHTS_DIR, model_name)
        return tf.keras.models.load_model(model_path)

    @classmethod
    def create_policy_dir_if_needed(cls):
        FileWorker.create_if_not_exist(POLICY_WEIGHTS_DIR)


if __name__ == '__main__':
    print(KerasLogger.logging_name(CAR_DATASET_PATH))
    print(KerasLogger.logging_path(CAR_DATASET_PATH))
    model = KerasLogger.load_latest_policy()
    model.summary()
