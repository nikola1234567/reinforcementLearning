from enum import Enum

import numpy as np
from sklearn import model_selection
from tensorflow.keras.utils import to_categorical

from Apstractions.DataPreprocessing.DataEncoders import *
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import *
from Apstractions.FileApstractions.FileWorker import FileWorker


def sting_to_integer(s):
    """
    Konverzija na string vo integer
    :param s: string
    :return: integer
    """
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n


def unique(list_array):
    x = np.array(list_array)
    return np.unique(x)


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

    def classes(self, result_type=ResultType.ENCODED):
        classes = self.dataset(result_type=result_type) \
            .groupby(self.classes_names(result_type=result_type)) \
            .size() \
            .reset_index() \
            .rename(columns={0: 'count'})
        classes = classes.drop('count', axis=1)
        return DataFrameWorker.row_list(classes)

    def complex_type_features(self):
        """
        :param data: data frame of the dataset
        :return: size of feature with complex type
        """
        return None


class ImageDataSet(Dataset):

    def __init__(self, absolute_path, delimiter=","):
        super(ImageDataSet, self).__init__(absolute_path, delimiter)
        self.processed_data = self.fer_dataset()

    def complex_type_features(self):
        """
        :param data: data frame of the dataset
        :return: size of feature with complex type
        """
        # for col in self.dataset_df.columns:
        #     tmp = col[0]
        #     if len(tmp) > 0:
        #         tmp1 = tmp[0]
        #         if len(tmp1) > 0:
        #             return len(tmp), len(tmp1)
        # return 0, 0
        return 48, 48, 1

    def fer_dataset(self):
        """helper function for image dataset"""

        list_class_matrix = list()
        for i in range(1500):
            mat = np.zeros((48, 48), dtype=np.uint8)
            txt = self.dataset_df['pixels'][i]
            ex_class = self.dataset_df['emotion'][i]
            # sekoj pixel kako string
            words = txt.split()
            # 2304 vrednosti vo sekoj red bidejki slikite se so dimanzii 48x48
            for j in range(2304):
                # floor division za dobivanja na indeksot x (redicata)
                xind = j // 48
                # module za dobivanje na indeksot y (kolonata)
                yind = j % 48
                # smestuvanje na vrednosta od nizata na soodvetno mesto vo matricata
                mat[xind][yind] = sting_to_integer(words[j])
            list_class_matrix.append((ex_class, mat.reshape((48, 48))))
        return list_class_matrix

    def classes_names(self, result_type=ResultType.ENCODED):
        classes = [elem[0] for elem in self.processed_data]
        return unique(classes)

    def split_feature_classes_image(self, data, result_type=ResultType.ENCODED):
        features = [elem[1] for elem in data]
        classes = [elem[0] for elem in data]
        classes = np.asarray(classes).astype(np.float32)
        feature_array = np.array(features)
        feature_array.reshape((len(features), 48, 48, 1))
        encoded_classes = to_categorical(classes)
        return feature_array, encoded_classes

    def split_data(self, result_type=ResultType.ENCODED, train_size=0.7, test_size=0.3):
        process_data = self.processed_data
        train_data = process_data[:int((100 - 30) / 100 * len(process_data))]
        test_data = process_data[int((100 - 30) / 100 * len(process_data)):]

        train_data_features, train_data_classes = self.split_feature_classes_image(data=train_data)
        test_data_features, test_data_classes = self.split_feature_classes_image(data=test_data)
        return train_data_features, train_data_classes, test_data_features, test_data_classes, train_data, test_data

    def classes(self, result_type=ResultType.ENCODED):
        if result_type == ResultType.PLAIN:
            return self.classes_names()
        return to_categorical(self.classes_names())


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/Car.csv "
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
    print(f'Classes unique ENCODED\n {dataset.classes()}')
    print(f'Classes unique PLAIN\n {dataset.classes(result_type=ResultType.PLAIN)}')

    dataset = ImageDataSet(absolute_path=FER_2013_PATH)
    print(dataset.classes())
    print(dataset.classes(ResultType.PLAIN))
