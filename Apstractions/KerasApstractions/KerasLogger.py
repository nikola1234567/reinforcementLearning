from datetime import datetime
import os
from configurations import POLICY_WEIGHTS_DIR
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import CAR_DATASET_PATH
from Apstractions.FileApstractions.FileWorker import FileWorker


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
        location = KerasLogger.logging_path(dataset_path)
        network_model.save(location)

    @classmethod
    def clean_policy_log_history(cls):
        FileWorker.clean_directory(POLICY_WEIGHTS_DIR)


if __name__ == '__main__':
    print(KerasLogger.logging_name(CAR_DATASET_PATH))
    print(KerasLogger.logging_path(CAR_DATASET_PATH))
