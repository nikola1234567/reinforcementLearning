import os

# ROOT PROJECT DIRECTORY PATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# POLICY WEIGHTS DIRECTORY
policy_weights_directories = '{},RLScripts,PolicyWeights'.format(ROOT_DIR).split(",")
POLICY_WEIGHTS_DIR = os.path.join(*policy_weights_directories)

# POLICY LOGS DIRECTORY
policy_logs_directories = '{},RLScripts,TensorBoardLogs'.format(ROOT_DIR).split(",")
POLICY_LOGS_DIR = os.path.join(*policy_logs_directories)

# POLICY EPOCH TRACKER
policy_epoch_tracker = '{},RLScripts,epoch_tracker.txt'.format(ROOT_DIR).split(",")
POLICY_EPOCH_TRACKER = os.path.join(*policy_epoch_tracker)

# LOGS DIR
tensorboard_logs_dir = '{},TensorBoard,Logs'.format(ROOT_DIR).split(",")
TENSORBOARD_LOGS_DIR = os.path.join(*tensorboard_logs_dir)