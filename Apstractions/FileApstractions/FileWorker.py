import ntpath
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import CAR_DATASET_PATH
import os
import shutil
from configurations import POLICY_WEIGHTS_DIR


class FileWorker:

    @classmethod
    def full_file_name(cls, file_path):
        return ntpath.basename(file_path)

    @classmethod
    def file_name(cls, file_path):
        full_name = FileWorker.full_file_name(file_path)
        return full_name.split(".")[0]

    @classmethod
    def clean_directory(cls, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @classmethod
    def content_of_directory(cls, directory_path):
        return os.listdir(directory_path)

    @classmethod
    def create_if_not_exist(cls, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)


if __name__ == '__main__':
    print(FileWorker.full_file_name(CAR_DATASET_PATH))
    print(FileWorker.file_name(CAR_DATASET_PATH))
    print(FileWorker.content_of_directory(POLICY_WEIGHTS_DIR))
