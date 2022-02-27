import os

# ROOT PROJECT DIRECTORY PATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# POLICY WEIGHTS DIRECTORY
policy_weights_directories = '{},RLScripts,PolicyWeights'.format(ROOT_DIR).split(",")
POLICY_WEIGHTS_DIR = os.path.join(*policy_weights_directories)

# SVM FEATURES DIRECTORY
svm_features_directories = '{},SVM,Features,features.csv'.format(ROOT_DIR).split(",")
SVM_FEATURES_DIR = os.path.join(*svm_features_directories)

# SVM CLASSES DIRECTORY
svm_classes_directories = '{},SVM,Classes'.format(ROOT_DIR).split(",")
SVM_CLASSES_DIR = os.path.join(*svm_classes_directories)