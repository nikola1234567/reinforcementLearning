import os

from configurations import ROOT_DIR


def self():
    self_path = os.path.join(ROOT_DIR, "Apstractions")
    self_path = os.path.join(self_path, "DatasetApstractions")
    self_path = os.path.join(self_path, "DatasetSamples")
    return self_path


SELF_PATH = self()

CAR_DATASET_PATH = os.path.join(SELF_PATH, "Car.csv")
POKEMON_DATASET_PATH = os.path.join(SELF_PATH, "Pokemon.csv")
FER_2013_PATH = os.path.join(SELF_PATH, "fer2013.csv")
MOBILE_PHONES_DATASET_PATH = os.path.join(SELF_PATH, "MobilePhones.csv")
HEART_FAILURE_DATASET_PATH = os.path.join(SELF_PATH, "Heart.csv")
WORLD_METROS_DATASET_PATH = os.path.join(SELF_PATH, "WorldMetros.csv")
