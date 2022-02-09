from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler
from Apstractions.DataPreprocessing.PandasApstractions import DataFrameWorker
import pandas as pd


class EncoderLogger:
    def __init__(self):
        self.column_header_value_pairs = {}
        self.column_value_header_pairs = {}
        self.column_value_name_value_pairs = {}
        self.column_value_value_name_pairs = {}

    def log_new_column(self, old_column, new_columns):
        self.column_header_value_pairs[old_column] = new_columns
        self.column_value_header_pairs[new_columns] = old_column

    def log_new_values(self, new_values):
        for value, encoding_value in zip(new_values, range(1, len(new_values) + 1)):
            self.column_value_name_value_pairs[value] = encoding_value
            self.column_value_value_name_pairs[encoding_value] = value

    def log(self, old_column, new_columns):
        self.log_new_column(old_column, new_columns)
        self.log_new_values(new_columns)

    def column_header_value(self, column_name):
        return self.column_header_value_pairs[column_name]

    def column_header_name(self, column_value):
        return self.column_value_header_pairs[column_value]

    def column_value_vector(self, column_value):
        return self.column_value_name_value_pairs[column_value]

    def column_vector_value(self, column_vector):
        return self.column_value_value_name_pairs[column_vector]


class Encoders:

    @classmethod
    def encode(cls, data):
        logger = EncoderLogger()
        categorical_columns = DataFrameWorker.categorical_columns(data)
        categorical_columns_indexes = DataFrameWorker.map_columns_to_index(data, categorical_columns)
        for column_index, column_name in zip(categorical_columns_indexes, categorical_columns):
            column_names = data[column_name].unique().tolist()
            name_dict, name_list = Encoders.new_names(column_names, column_name)
            logger.log(column_name, tuple(name_list))
            one_hot = pd.get_dummies(data[column_name])
            one_hot = one_hot.rename(name_dict, axis=1)
            data = data.drop(column_name, axis=1)
            data = data.join(one_hot)
        return data, logger

    @classmethod
    def new_names(cls, names, prefix):
        dict = {name: prefix + "_" + name for name in names}
        list_name = [prefix + "_" + name for name in names]
        list_name.sort()
        return dict, list_name


if __name__ == '__main__':
    # values = ['bad', 'good']
    # print(Encoders.one_hot_encoder(values))

    print("=============== Label Encoder ===============\n")
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/car.csv "
    dataframe = CSVFileHandler(datasetPath).df()
    # print(Encoders.label_encoder(dataframe))
    # encoded = Encoders.one_hot_encoder(dataframe)
    # print(encoded)
    # print("\n\n==============================================================\n\n")
    Encoders.encode(dataframe)
