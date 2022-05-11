# Neural Architecture Search with Deep Reinforcement Learning

## Motivation

## REINFORCE Algorithm

As previously stated the core of this project is to show the power of reinforcement learning, more specifically deep
reinforcement learning. That is a combination of the most currently relevant AI fields which are Deep Learning and
Reinforcement learning. Deep learning is one of the most useful tools because it can quicly adapt to every kind of problem
nature, from most complex to most simple, as long as large amount of data is given to them, so they can learn, whilst other
ML learning models can behave well only in certain environments, under certain conditions. However, from there we can deduct
a crucial point when using neural networks. That is no matter how good they are  and how fast they adapt, that is only possible
if large amount of data is given.

The deep reinforcement learning field basically can be divided in two types of algorithms, and those are `policy-based` and
`policy-gradient` methods. 

![deep reinforcement learning algorithms](./images/REINFORCE/deep-reinforcement-learning-algorithms.png
)

## State

A state is represented by a set of parameters for the neural network, to create a neural network suitable for the given
dataset. The parameters included in the state are state(num_classes, num_features, num_layers, hidden_size,
learning_rate, conv_size)

## Action

The actions in our project are represented as a set of changeable parameters for the neural network. The representation
is like the state, but the actions are constructed just of the parameters that can be changed and the performance is
dependent on them in the neural network. Every next possible action has a change in just one of the parameters from the
previous, consequently the number of the next possible actions from a current state is the number of parameters in the
action.

## Generator

The generator is providing the neural network models contracted by the current state updated in the current action.
Depending on the scale of the dataset and its features the generator provides a suitable model whose performance is
rewarded by the success of the training phase. The generator has two methods generating models one that generates feed
forward neural networks and the other one generates convolutional neural networks.

```sh
    def model_from_state(self, state):
    """
        Method generating a feed froward sequential neural network with the parameter from the state
        :param state - object from class State
        :return sequential keras model
    """
    
     def model_conv_from_state(self, state):
        """
        Method generating a convolutional neural network with the parameter from the state
        :param state - object from class State
        :return sequential keras model
        """


```

## Controller

The controller is the main part in which all the steps are taken and the logic is implemented. All parts of the
application are coming together and eventually communicating through its actions. Firstly, the controller uses all the
other classes to initialize its initial class parameters and

## Data preprocessing

So far the project only works with standard datasets and image datasets. We created class abstractions in order to
process datasets. In the file `DatasetApstractions` there's a class called `Dataset` which only serves to process **
standard datasets** i.e. dataset in CSV format, which only consists of plain features and target class (whether is
multiple columns or not). Only input parametrs are the absolute path to where the dataset is stored, as well as the
delimiter used in the CSV file. A requirement for our `Dataset` class to works is that every column in the dataset
should be prefixed with `class_<column-name>` (ex. class_humidity), in order to recognize them when working with them.
Then for every dataset we have several methods out of the box:

```sh
    def number_of_features(self, result_type=ResultType.ENCODED)
	"""
        returns the number of features in the dataset specified with the result type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: number of features (type - Int)
    """

    def number_of_classes(self, result_type=ResultType.ENCODED)
	"""
        number of columns in the dataset specified with result_type which represent the target classes
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: number of classes (type - Int)
    """

    def classes_names(self, result_type=ResultType.ENCODED)
	"""
        name of the columns which are the target classes in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of names (type - List)
    """

    def feature_names(self, result_type=ResultType.ENCODED)
	"""
        returns the feature name in the dataset specified with the result_type
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :return: list of feature names (type - List)
    """

    def split_data(self, result_type=ResultType.ENCODED, train_size=0.7, test_size=0.3)
	"""
        Splits the dataset specified with the result_type in 6 sets of values:
            1. train_data - default 70% from the dataset
            2. test_data - default 30% from the dataset
            3. train_data_features - only the features in train_data
            4. train_data_classes - only the classes in train_data
            5. test_data_features - only the features in test_data
            6. test_data_classes - only the classes in test_data
        :param result_type: Should the method be applied on the encoded dataset or plain dataset (type - ResultType)
        :param train_size: Percentage of dataset for training (default = 0.7 - 70%, type - Decimal)
        :param test_size: Percentage of dataset for testing (default = 0.3 - 30%, type - Decimal)
        :return: train_data_features, train_data_classes, test_data_features, test_data_classes, train_data, test_data
        (in the specified order, type - DataFrame)
    """
```
Those are some of the most important and many others. For example one of the most useful are `number_of_classes` and `number_of_features` in order
to build compatible ***Input*** and ***Output*** layer of the Neural Network model we are about to build.
As you can notice each method receives a parameter called result type which can have two possible values:
```sh
class ResultType(Enum):
    ENCODED = 1
    PLAIN = 2
```
When we process the dataset we keep two versions of the dataset one PLAIN(as it is) and one ENCODED with one hot encoding where every categorical value
is encoded accordingly.
Previously i mentioned that we also support image datasets but for them there is no generic way to apstract them with one class, the way we are handling them 
is we derive the `Dataset` class and implement each method one by one to be able to process the concrete dataset.

## Policy

The policy state is kept throughout the time. With the help of `Keras`, since the policy is a neural network we save the policy weights after each dataset,
so on the next dataset it can load the weights first and then act with greater knowledge.
```sh
.../RLPolicyAgent.py

    def load_model(self):
        try:
            return KerasLogger.load_latest_policy()
        except PolicyWeightsNotFound:
            return self._build_model()
```

## NAS Environment

This is our abstraction of the standard `Gym` environments in order to be able to treat the problem as a standard reinforcement learning problem.
It offers the well-known `step` method which returns the know variables but with our computation logic:
```sh
		1. state - the model which was trained
		2. reward - accuracy on the specified dataset
		3. done - if the rewards starts to constantly decrease (currently done after 3 iterations) - NEEDS TO BE IMPROVED
		4. info - dictionary of two fields:
			-rewards during playing (so far)
			-taken actions (so far)
```

## Tensorboard visualisations

`Tensorboard` is the tool we decided to use for visualisations. So far we save the following statistics:

1. Accuracy/ Loss/ Mean square error
These are regarding the metrics while `training` of the proposed model.

![accuracy](./images/1_accuracy.png)
![loss](./images/1_loss.png)
![mse](./images/1_mse.png)

2. Confusion matrix
We keep track of each model performance our algorithm proposes on the testing portion of the dataset, so we can
track the progress over time.

![](./images/2_confusion_matrix.png)

3. Histograms & distributions of neural network weights and biases

![distributions](./images/3_distributions.png)
![histograms](./images/3_histograms.png)

4. Hyper parameters 
This helps to see which combinations of our actions resulted in the best accuracy on the `test` dataset, not while
training as in the first point which is while `training`.

![table view](./images/4_1_hparam.png)
![parallel coordinates view](./images/4_2_hparam.png)

5. Detailed projector reports
Here we have different reports like what was more exhausting on hardware, what should we do to improve our model,
what part of the neural network is most exhausting the algorith etc.

![step time graph](./images/step_time_graph.png)
![tensorflow stats](./images/stats.png)
![recommendations](./images/reccomendations.png)

More details about concrete visualisations and their evaluation in the section about testing phases.