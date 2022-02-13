from enum import Enum
from sklearn import model_selection
from Apstractions.DataPreprocessing.DataEncoders import *


class ResultType(Enum):
    ENCODED = 1
    PLAIN = 2


class Dataset:

    def __init__(self, absolute_path, delimiter=","):
        self.absolutePath = absolute_path
        self.csv_handler = CSVFileHandler(self.absolutePath, delimiter=delimiter)
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
        number of unique values for the classes in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: number of classes (type - Int)
        """
        return len(self.classes_names(result_type))

    def classes_names(self, result_type=ResultType.ENCODED):
        """
        name of the columns which are the target classes in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of names (type - List)
        """
        working_dataset = self.dataset(result_type=result_type)
        return filter(lambda name: name.startswith('class_'), list(working_dataset.columns))

    def feature_names(self, result_type=ResultType.ENCODED):
        """
        returns the feature name in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of feature names (type - List)
        """
        working_dataset = self.dataset(result_type=result_type)
        return list(filter(lambda element: element not in self.classes_names(result_type=result_type),
                           list(working_dataset.columns)))

    def classes(self, result_type=ResultType.ENCODED):
        """
        finds the unique values for the class property in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of class values (type - List)
        """
        if result_type == ResultType.ENCODED:
            return self.encoded_classes()
        return self.plain_classes()

    def encoded_classes(self):
        """
        :return: unique classes in the encoded dataset (type - List)
        """
        class_columns = list(self.encoded_dataset_logger.column_header_value(self.target_class_label))
        class_columns_dataset = self.encoded_dataset_df[class_columns]
        unique_rows = class_columns_dataset.drop_duplicates(subset=class_columns)
        return unique_rows.values.tolist()

    def plain_classes(self):
        """
        :return: unique classes in the plain dataset (type - List)
        """
        return list(self.dataset_df[self.target_class_label].unique())

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
                                                                 stratify=node_data[self.class_label(result_type)])
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
        c_labels = self.class_label(result_type)
        features = data[[column for column in data.columns if column not in c_labels]]
        classes = data[c_labels]
        return features, classes

    def class_encoded_dict(self):
        # TODO: Needs logger refactor first, currently not correct!!
        c_labels = self.classes()
        dictionary = {label: self.encoded_dataset_logger.column_header_value(tuple(label)) for label in c_labels}
        return dictionary


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/car.csv "
    dataset = Dataset(datasetPath, "acceptability")
    print(f'Feature names ENCODED: {dataset.feature_names()}')
    print(f'Feature names PLAIN: {dataset.feature_names(ResultType.PLAIN)}')
    print(f'Number of features ENCODED: {dataset.number_of_features()}')
    print(f'Number of features PLAIN: {dataset.number_of_features(ResultType.PLAIN)}')
    # print(f'Classes ENCODED: {dataset.classes()}')
    # print(f'Classes PLAIN: {dataset.classes(ResultType.PLAIN)}')
    print(f'Number of classes ENCODED: {dataset.number_of_classes()}')
    print(f'Number of classes PLAIN: {dataset.number_of_classes(ResultType.PLAIN)}')
    # train_f, train_c, test_f, test_c, train, test = dataset.split_data()
    print("====================")
