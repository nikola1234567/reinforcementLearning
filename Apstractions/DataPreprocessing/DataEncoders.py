from sklearn.preprocessing import OneHotEncoder
from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler


class Encoders:

    """source: https://towardsdatascience.com/encoding-categorical-features-21a2651a065c#:~:text=OneHotEncoder%20has
    %20the%20option%20to,and%20support%20sparse%20matrix%20output.&text=But%20if%20the%20number%20of,
    it%20supports%20sparse%20matrix%20output. """

    @classmethod
    def one_hot_encoder(cls, value_list):
        return OneHotEncoder().fit_transform(value_list).toarray()

    @classmethod
    def label_encoder(cls):
        pass

    @classmethod
    def dict_vectorizer(cls, data):
        # data -> dataframe
        # Categorical boolean mask
        categorical_feature_mask = data.dtypes == object
        # filter categorical columns using mask and turn it into a list
        categorical_cols = data.columns[categorical_feature_mask].tolist()
        return categorical_cols


if __name__ == '__main__':
    # values = ['bad', 'good']
    # print(Encoders.one_hot_encoder(values))

    print("=============== DICT VECTORIZER ===============\n")
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/wine_quality.csv "
    dataframe = CSVFileHandler(datasetPath).df()
    print("Not encoded \n=====================================================\n\n")
    print(dataframe)
    print("=====================================================\n\n")
    print("Encoded \n=====================================================\n\n")
    print(Encoders.dict_vectorizer(dataframe))
    print("=====================================================\n\n")
