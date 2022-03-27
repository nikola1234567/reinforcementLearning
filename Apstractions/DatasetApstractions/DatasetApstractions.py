from enum import Enum
from sklearn import model_selection
from Apstractions.DataPreprocessing.DataEncoders import *
from Apstractions.FileApstractions.FileWorker import FileWorker
import numpy as np


class ResultType(Enum):
    ENCODED = 1
    PLAIN = 2


class Dataset:

    def __init__(self, absolute_path, delimiter=","):
        self.absolute_path = absolute_path
        self.csv_handler = CSVFileHandler(self.absolute_path, delimiter=delimiter)
        self.dataset_df = self.csv_handler.df()
        encoded = Encoders.encode(self.dataset_df)
        self.encoded_dataset_df = encoded

    def dataset(self, result_type=ResultType.ENCODED):
        """
        returns the dataset based on the result type we desire
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: dataset (type - DataFrame)
        """
        if result_type == ResultType.ENCODED:
            return self.encoded_dataset_df
        return self.dataset_df

    def number_of_features(self, result_type=ResultType.ENCODED):
        """
        returns the number of features in the dataset specified with the result type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: number of features (type - Int)
        """
        return len(self.feature_names(result_type))

    def number_of_classes(self, result_type=ResultType.ENCODED):
        """
        number of columns in the dataset specified with result_type which represent the target classes
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: number of classes (type - Int)
        """
        return len(self.classes_names(result_type=result_type))

    def classes_names(self, result_type=ResultType.ENCODED):
        """
        name of the columns which are the target classes in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of names (type - List)
        """
        working_dataset = self.dataset(result_type=result_type)
        return list(filter(lambda name: name.startswith('class_'), list(working_dataset.columns)))

    def feature_names(self, result_type=ResultType.ENCODED):
        """
        returns the feature name in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of feature names (type - List)
        """
        working_dataset = self.dataset(result_type=result_type)
        classes_names = self.classes_names(result_type=result_type)
        return list(filter(lambda element: element not in classes_names,
                           list(working_dataset.columns)))

    def split_data(self, result_type=ResultType.ENCODED, train_size=0.7, test_size=0.3):
        """
        Splits the dataset specified with the result_type in 6 sets of values:
            1. train_data - default 70% from the dataset
            2. test_data - default 30% from the dataset
            3. train_data_features - only the features in train_data
            4. train_data_classes - only the classes in train_data
            5. test_data_features - only the features in test_data
            6. test_data_classes - only the classes in test_data
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :param train_size: Percentage of dataset for training (default = 0.7 - 70%, type - Decimal)
        :param test_size: Percentage of dataset for testing (default = 0.3 - 30%, type - Decimal)
        :return: train_data_features, train_data_classes, test_data_features, test_data_classes, train_data, test_data
        (in the specified order, type - DataFrame)
        """
        node_data = self.dataset(result_type)
        train_data, test_data = model_selection.train_test_split(node_data,
                                                                 train_size=train_size,
                                                                 test_size=test_size,
                                                                 stratify=node_data[self.classes_names(result_type)])
        train_data_features, train_data_classes = self.split_feature_classes(train_data)
        test_data_features, test_data_classes = self.split_feature_classes(test_data)
        return train_data_features, train_data_classes, test_data_features, test_data_classes, train_data, test_data

    def split_feature_classes(self, data, result_type=ResultType.ENCODED):
        """
        Splits the given data in features and classes
        :param data: Data that will be split (type - DataFrame)
        :param result_type:  Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: Splitted data.
        """
        c_labels = self.classes_names(result_type)
        features = data[[column for column in data.columns if column not in c_labels]]
        features = np.asarray(features).astype(np.float32)
        classes = data[c_labels]
        classes = np.asarray(classes).astype(np.float32)
        return features, classes

    def name(self):
        return FileWorker.file_name(self.absolute_path)


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/car.csv "
    dataset = Dataset(datasetPath)
    print(f'Feature names ENCODED: {dataset.feature_names()}')
    print(f'Feature names PLAIN: {dataset.feature_names(ResultType.PLAIN)}')
    print(f'Number of features ENCODED: {dataset.number_of_features()}')
    print(f'Number of features PLAIN: {dataset.number_of_features(ResultType.PLAIN)}')
    print(f'Classes names ENCODED: {dataset.classes_names()}')
    print(f'Classes names PLAIN: {dataset.classes_names(ResultType.PLAIN)}')
    print(f'Number of classes ENCODED: {dataset.number_of_classes()}')
    print(f'Number of classes PLAIN: {dataset.number_of_classes(ResultType.PLAIN)}')
    train_f, train_c, test_f, test_c, train, test = dataset.split_data()
    print("====================")
