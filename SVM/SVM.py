from sklearn.svm import SVC
import uuid
from Apstractions.FileApstractions.CSVApstractions import CSVFileHandler
from configurations import SVM_FEATURES_DIR, SVM_CLASSES_DIR
from Apstractions.KerasApstractions.KerasLogger import KerasLogger
from NAS.Generator import Generator
from NAS.State import State
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import CAR_DATASET_PATH
from Apstractions.DatasetApstractions.DatasetApstractions import Dataset

GAUSSIAN_KERNEL = 'rbf'


class SVM:
    def __init__(self):
        KerasLogger.create_network_directory_if_needed(SVM_CLASSES_DIR)
        self.classifier = SVC(kernel=GAUSSIAN_KERNEL)
        self.csv_handler = CSVFileHandler(file_path=SVM_FEATURES_DIR)

    def add_feature(self, dataset_configuration, network):
        network_name = SVM.network_name()
        self.add_configuration(dataset_configuration, network_name)
        SVM.add_network(network, network_name)

    def add_configuration(self, dataset_configuration, network_name):
        full_data = dataset_configuration.executable_configuration()
        full_data.append(network_name)
        self.csv_handler.add_row(list_data=full_data)

    @staticmethod
    def add_network(network, network_name):
        KerasLogger.save_custom_network(location_path=SVM_CLASSES_DIR,
                                        network_name=network_name,
                                        network=network)

    @staticmethod
    def network_name():
        return str(uuid.uuid4())

    @staticmethod
    def clean_svm_classes():
        KerasLogger.clean_network_directory(SVM_CLASSES_DIR)


if __name__ == '__main__':
    # SVM.clean_svm_classes()
    generator = Generator()
    dataset = Dataset(absolute_path=CAR_DATASET_PATH)
    state = State(dataset.number_of_classes(), dataset.number_of_features(), 1, 1, 0.5)
    network = generator.model_from_state(state=state)
    svm = SVM()
    svm.add_feature(dataset_configuration=dataset.configuration(),
                    network=network)

