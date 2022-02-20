import pandas as pd


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
