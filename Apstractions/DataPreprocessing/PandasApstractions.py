from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler


class DataFrameWorker:

    @classmethod
    def numerical_columns(cls, df):
        """

        :param df: dataframe
        :return: list of columns
        """
        return list(df._get_numeric_data().columns)

    @classmethod
    def categorical_columns(cls, df):
        """

        :param df: dataframe
        :return: list of columns
        """
        num_columns = df._get_numeric_data().columns
        return list(set(df.columns) - set(num_columns))

    @classmethod
    def map_columns_to_index(cls, df, columns):
        """
            Column mapper
        Map's column names provided in `columns` parameter
        to their corresponding index.
        :param df: dataframe
        :param columns: columns list
        :return: list of indexes
        """
        return [df.columns.get_loc(elem) for elem in columns]


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/wine_quality.csv "
    dataframe = CSVFileHandler(datasetPath, delimiter=";").df()
    print(DataFrameWorker.categorical_columns(dataframe))
    print(DataFrameWorker.map_columns_to_index(dataframe, ['quality', 'alcohol']))
