import os

# ROOT PROJECT DIRECTORY PATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# POLICY WEIGHTS DIRECTORY
policy_weights_directories = '{},RLScripts,PolicyWeights'.format(ROOT_DIR).split(",")
POLICY_WEIGHTS_DIR = os.path.join(*policy_weights_directories)