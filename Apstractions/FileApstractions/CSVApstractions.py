import pandas as pd
from keras.callbacks import CSVLogger
from datetime import datetime
import os

from configurations import NAS_ENVIRONMENT_LOG_DIR

now = datetime.now()


def csv_logger_callback(dataset_name):
    file_name = '{}_{}'.format(dataset_name, now.strftime("%m_%d_%Y_%H_%M"))
    file_path = os.path.join(NAS_ENVIRONMENT_LOG_DIR, file_name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    file_path = os.path.join(file_path, '{}.csv'.format(file_name))
    return CSVLogger(file_path, append=True, separator=';')


class CSVFileHandler:

    def __init__(self, file_path, delimiter=","):
        self.filePath = file_path
        self.file_dataframe = pd.read_csv(file_path, delimiter=delimiter)

    def df(self):
        return self.file_dataframe

    def number_of_fields(self):
        return self.file_dataframe.shape[1]

    def statistics(self):
        # TODO: TO BE DONE
        return self.file_dataframe.describe()


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/wine_quality.csv "
    csvHandler = CSVFileHandler(datasetPath)
    print(f'Number of fields {csvHandler.number_of_fields()}')
