import numpy as np
import pickle
import pandas as pd
import nn_lib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Activation,Dropout
import tensorflow.keras as keras
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.constraints import max_norm
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
import datetime


class ClaimClassifier:
    def __init__(self,epochs=30,batch_size=32):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        #self.claimNN = nn_lib.MultiLayerNetwork(input_dim, neurons, activations)
        self.epochs = epochs
        self.batch_size = batch_size
        optimizer = 'SGD'
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=9))
        self.model.add(Dense(units=32,activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))


        self.model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy']) 

        self.predition = np.array([])


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

        return  X

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
        """
        self.model.fit(X_clean,y_raw, epochs=self.epochs, batch_size=self.batch_size)

        """
        trainer = nn_lib.Trainer(
            network=self.claimNN,
            batch_size=8,
            nb_epoch=1000,
            learning_rate=0.01,
            loss_fun="cross_entropy",
            shuffle_flag=True,
        )
        trainer.train(X_clean,y_raw)
        """

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
        # YOUR CODE HERE

        preds = self.model.predict(X_raw, batch_size=self.batch_size)
        self.predition = preds
        #preds = self.claimNN(X_clean).argmax(axis=1).squeeze()
        return  preds# YOUR NUMPY ARRAY


    def evaluate_architecture(self,x_val,y_val,predicted_y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        ############################
        #1.Plot ROC-AUC curve
        #2.Print out evaluate loss and accuracy
        #############################
        plot_roc(y_val,predicted_y)
        fpr, tpr, threshold = metrics.roc_curve(y_val, predicted_y)
        auc = metrics.auc(fpr, tpr)

        best_threshold = threshold[np.argmax(tpr - fpr)]
        y_pred=(predicted_y>best_threshold) #classify to label
        ########################
        print()
        loss_and_metrics = self.model.evaluate(x_val, y_val)
        print("Loss is %.2f and accuracy is %.2f"%(loss_and_metrics[0],loss_and_metrics[1]))
        return auc

    def save_model(self):
        with open("part2_claim_classifier.pickle", "wb") as target:
            pickle.dump(self, target)


"""Grid search for hyper parameters and optimizers ranked by auc score"""
def ClaimClassifierHyperParameterSearch(x_train,y_train, estimator, param_grid):
    #print the training start time
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))


    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,refit='roc_auc',scoring=['roc_auc','accuracy'])
    grid_result = grid.fit(x_train,y_train) 
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    

    #print the time when finished training
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

    return grid_result.best_params_

def plot_roc(y_val,predicted_y):

    fpr, tpr, threshold = metrics.roc_curve(y_val, predicted_y)
    roc_auc = metrics.auc(fpr, tpr)
    #best_threshold = threshold[np.argmax(tpr - fpr)]
    
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def create_model(optimizer = 'SGD'):

    
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=9))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy']) 
    return model
##################################################
##########Test code: Uncomment if you want test the code
"""
data = np.genfromtxt('part2_data.csv',delimiter=',')
a = ClaimClassifier(30,32)
split_index = int(0.8*data.shape[0])
#process_input= a._preprocessor(data[1:,:9])

y_train = data[1:split_index,-1]
x_train = process_input[:split_index-1,:]
print(x_train.shape)
print(y_train.shape)
x_val = process_input[split_index-1:,:]
y_val = data[split_index:,-1]

a.fit(x_train,y_train)
predicted_y = a.predict(x_val)
print("AUC score is %.3f"%a.evaluate_architecture(x_val,y_val,predicted_y))
"""
#######################################################
## Parameter tuning---------Grid Search 
"""
optimizer = [ 'SGD', 'Adam' ]
# grid search epochs, batch size
epochs = [10,30,50]
batch_size = [4,8,16,32]
#batch_size = [32]
param_grid = dict(epochs=epochs, batch_size=batch_size,optimizer=optimizer)
#param_grid = dict(epochs=epochs, batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,refit='roc_auc',scoring=['roc_auc','accuracy'])
grid_result = grid.fit(x_train,y_train) 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(grid_result.cv_results_)
##############################################################
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
"""
##################################################
