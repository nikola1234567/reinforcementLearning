import pandas as pd


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

    @classmethod
    def decreasing_or_constant(cls, dataframe):
        """
        A series is increasing/decreasing if more than 80% of the differences between
        consecutive values are strictly increasing/decreasing.
        :param dataframe: dataframe object with one column containing the series
        :return: True if decreasing or constant, False otherwise
        """
        diff = dataframe[0] - dataframe[0].shift(1)
        # if less than 20% are smaller than their successor the series is increasing
        increasing = (diff <= 0).sum() <= 0.2 * len(dataframe)
        if increasing:
            return False
        # if less than 20% are bigger than their successor the series is decreasing
        decreasing = (diff >= 0).sum() <= 0.2 * len(dataframe)
        if decreasing:
            return True
        if not increasing and not decreasing:
            return True

    @staticmethod
    def row_list(dataframe):
        rows = [','.join(str(x) for x in elem) for elem in dataframe.values]
        return rows


if __name__ == '__main__':
    df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(DataFrameWorker.decreasing_or_constant(df))
    df = pd.DataFrame([10, 11, 12, 13, 10, 6, 7, 7, 6, 5])
    print(DataFrameWorker.decreasing_or_constant(df))
    df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(DataFrameWorker.decreasing_or_constant(df))
    df = pd.DataFrame({'Date': ['10/2/2011', '11/2/2011', '12/2/2011', '13/2/11'],
                       'Event': ['Music', 'Poetry', 'Theatre', 'Comedy'],
                       'Cost': [10000, 5000, 15000, 2000]})
    print(DataFrameWorker.row_list(df))
