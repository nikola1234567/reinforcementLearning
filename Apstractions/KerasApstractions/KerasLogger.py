from datetime import datetime
import os
from configurations import POLICY_WEIGHTS_DIR


class KerasLogger:

    @classmethod
    def logging_name(cls, dataset_name):
        current_datetime = datetime.now()
        cd_string = current_datetime.strftime("%d_%m_%Y_%H_%M")
        return '{}_{}_dataset.h5'.format(cd_string, dataset_name)

    @classmethod
    def logging_path(cls, dataset_name):
        logging_name = KerasLogger.logging_name(dataset_name)
        return os.path.join(POLICY_WEIGHTS_DIR, logging_name)

    @classmethod
    def save_network(cls, network_model, dataset_name):
        location = KerasLogger.logging_path(dataset_name)
        network_model.save(location)


if __name__ == '__main__':
    print(KerasLogger.logging_name("car"))
    print(KerasLogger.logging_path("car"))
