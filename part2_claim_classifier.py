import numpy as np
import pickle
import pandas as pd
import nn_lib


class ClaimClassifier:
    def __init__(self,):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.data = np.genfromtxt('part2_data.csv',delimiter=',')
        # print(self.data)
        # np.read('part2_data.csv')
        input_dim = 10
        neurons = [16, 3]
        activations = ["relu", "identity"]
        self.claimNN = nn_lib.MultiLayerNetwork(input_dim, neurons, activations)

        pass

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        prepro = nn_lib.Preprocessor(X_raw)
        X = prepro.apply(X_raw)

        return  X# YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded
        y_raw : numpy.ndarray (optional)
            A one dimensional numpy array, this is the binary target variable

        Returns
        -------
        ?
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        X_clean = self._preprocessor(X_raw)
        # print(X_clean.shape)
        # print(y_raw.shape)
        trainer = nn_lib.Trainer(
            network=self.claimNN,
            batch_size=8,
            nb_epoch=1000,
            learning_rate=0.01,
            loss_fun="cross_entropy",
            shuffle_flag=True,
        )
        y_raw = y_raw.reshape(-1,1)
        print(y_raw.shape)
        trainer.train(X_clean,y_raw)
        pass

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        X_clean = self._preprocessor(X_raw)
        preds = self.claimNN(X_clean).argmax(axis=1).squeeze()
        return  preds# YOUR NUMPY ARRAY

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        pass

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


def ClaimClassifierHyperParameterSearch():  # ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters

a = ClaimClassifier()
split_index = int(0.8*a.data.shape[0])
x_train = a.data[1:split_index,0:10]
y_train = a.data[1:split_index,-1]
x_val = a.data[split_index:,0:10]
y_val = a.data[split_index:,-1]
a.fit(x_train,y_train)

predicted_y = a.predict(x_val)
# targets = y_val.argmax(axis=1).squeeze()
accuracy = (predicted_y == y_val).mean()
print(accuracy)