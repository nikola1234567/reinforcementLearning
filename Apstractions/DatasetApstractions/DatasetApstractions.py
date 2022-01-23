import sys
from sklearn import model_selection
from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler
from enum import Enum


class DatasetTypes(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 0


class Dataset:
    # TODO: Support for CONTINUOUS Dataset types

    def __init__(self, absolute_path, target_class_label, dataset_type=DatasetTypes.CATEGORICAL):
        self.dataset_type = dataset_type
        self.absolutePath = absolute_path
        self.target_class_label = target_class_label
        self.class_targets = {}
        self.csv_handler = CSVFileHandler(self.absolutePath, delimiter=";")

    def number_of_features(self):
        return self.csv_handler.number_of_fields() - 1

    def feature_names(self):
        data_frame = self.csv_handler.df()
        return list(filter(lambda element: element != self.target_class_label, list(data_frame.columns)))

    def number_of_classes(self):
        if self.dataset_type == DatasetTypes.CATEGORICAL:
            return len(self.classes())
        return sys.maxsize

    def classes(self):
        self.class_targets = self.csv_handler.df()[self.target_class_label].unique()
        return list(self.class_targets)

    def split_data(self, train_size=0.7, test_size=0.3):
        node_data = self.csv_handler.df()
        train_data, test_data = model_selection.train_test_split(node_data,
                                                                 train_size=train_size,
                                                                 test_size=test_size,
                                                                 stratify=node_data[self.target_class_label])
        return train_data, test_data

    def split_feature_classes(self, data):
        features = data.loc[:, data.columns != self.target_class_label]
        classes = data[self.target_class_label]
        return features, classes


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/wine_quality.csv "
    dataset = Dataset(datasetPath, "quality")
    print(f'Feature names: {dataset.feature_names()}')
    print(f'Number of features {dataset.number_of_features()}')
    print(f'Classes {dataset.classes()}')
    train_data, test_data = dataset.split_data()
    print(f'Train data \n=========================================== \n {len(train_data)} \n {train_data} \n'
          f'=========================================== \n')
    train_data_features, train_data_classes = dataset.split_feature_classes(train_data)
    print(
        f'Train data features\n=========================================== \n {len(train_data_features)} \n {train_data_features} \n '
        f'=========================================== \n')
    print(
        f'Train data classes\n=========================================== \n {len(train_data_classes)} \n {train_data_classes} \n'
        f'=========================================== \n')
    # print(f'Train data \n=========================================== \n {len(test_data)} \n {test_data} \n'
    #       f'===========================================')
