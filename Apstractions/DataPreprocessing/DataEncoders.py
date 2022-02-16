from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler
from Apstractions.DataPreprocessing.PandasApstractions import DataFrameWorker
import pandas as pd


class Encoders:

    @classmethod
    def encode(cls, data):
        categorical_columns = DataFrameWorker.categorical_columns(data)
        prefixes = {column: column for column in categorical_columns}
        return pd.get_dummies(data,
                              prefix=prefixes,
                              drop_first=False)


if __name__ == '__main__':
    # values = ['bad', 'good']
    # print(Encoders.one_hot_encoder(values))

    print("=============== Label Encoder ===============\n")
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/car.csv "
    datasetPath2 = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/wine_quality.csv "
    dataframe = CSVFileHandler(datasetPath).df()
    dataframe2 = CSVFileHandler(datasetPath2, delimiter=";").df()
    # print(Encoders.label_encoder(dataframe))
    # encoded = Encoders.one_hot_encoder(dataframe)
    # print(encoded)
    # print("\n\n==============================================================\n\n")
    df_encoded = Encoders.encode(dataframe)
    df_encoded2 = Encoders.encode(dataframe2)
    print(df_encoded.shape)
