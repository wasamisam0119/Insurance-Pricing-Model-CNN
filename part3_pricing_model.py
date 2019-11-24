from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import numpy as np
import pandas as pd
import nn_lib
from keras.models import Sequential
from keras.layers import Dense
import sklearn.metrics as metrics
from part2_claim_classifier import plot_roc
import tensorflow as tf


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False, preprocessor=None, epochs=10, batch_size=32, optimizer='Adam',
                 encoder=None, imputer=None, numerical=None, categorical=None):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.encoder = encoder
        self.imputer = imputer
        self.categorical = categorical
        self.numerical = numerical
        self.prepro = preprocessor
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_classifier = Sequential()
        self.base_classifier.add(Dense(units=128, activation='relu', input_dim=18))
        self.base_classifier.add(Dense(units=64, activation='relu'))
        self.base_classifier.add(Dense(units=1, activation='sigmoid'))
        self.base_classifier.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        columns = ['id_policy', 'pol_bonus', 'pol_coverage', 'pol_duration',
                   'pol_sit_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage',
                   'pol_insee_code', 'drv_drv2', 'drv_age1', 'drv_age2', 'drv_sex1',
                   'drv_sex2', 'drv_age_lic1', 'drv_age_lic2', 'vh_age', 'vh_cyl',
                   'vh_din', 'vh_fuel', 'vh_make', 'vh_model', 'vh_sale_begin',
                   'vh_sale_end', 'vh_speed', 'vh_type', 'vh_value', 'vh_weight',
                   'town_mean_altitude', 'town_surface_area', 'population', 'commune_code',
                   'canton_code', 'city_district_code', 'regional_department_code',
                   ]
        df = pd.DataFrame(data=X_raw, columns=columns)

        df2 = df[self.categorical].apply(self.encoder.fit_transform)

        # encoded_features = enc.transform(df_categorical).toarray()

        # encoded_features = pd.get_dummies(df[categorical_feature_names_sel]).to_numpy()
        # numerical = df[numerical_features_names_sel + categorical_feature_names_sel].to_numpy(dtype=np.float64)
        numerical = np.concatenate((self.imputer.transform(df[self.numerical].to_numpy(dtype=np.float64)),
                                    df2.to_numpy(dtype=np.float64)),
                                   axis=1)
        X = self.prepro.apply((numerical))
        # X = np.concatenate((numerical, encoded_features), axis=1)
        # inputs_df = pd.concat([df[numerical_features_names_sel], encoded_features], axis=1)
        return X  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            print(1)
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier.fit(X_clean, y_raw, epochs=self.epochs, batch_size=self.batch_size)
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        preds = self.base_classifier.predict(X_clean, batch_size=self.batch_size)
        return preds  # return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

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
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

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
        plot_roc(y_val, predicted_y)
        fpr, tpr, threshold = metrics.roc_curve(y_val, predicted_y)
        auc = metrics.auc(fpr, tpr)
        x_val = self._preprocessor(x_val)
        best_threshold = threshold[np.argmax(tpr - fpr)]
        # y_pred = (predicted_y > best_threshold)  #classify to label
        ########################
        loss_and_metrics = self.base_classifier.evaluate(x_val, y_val)
        print("Loss is %.2f and accuracy is %.2f"%(loss_and_metrics[0], loss_and_metrics[1]))
        return auc

    def save_model(self):
        self.base_classifier.save('part3_model.h5')  # creates a HDF5 file 'my_model.h5'
        del self.base_classifier

    def warp(self, new_model):
        self.base_classifier = new_model


def load_model():
    model = tf.keras.models.load_model('part3_model.h5')
    df = pd.read_csv('part3_data.csv').sample(frac=1)
    x_train = df[df.columns[:-2]].to_numpy()
    y_train = df[df.columns[-1]].to_numpy()
    claim_train = df[df.columns[-2]].to_numpy()
    numerical_features_names_sel = ['pol_bonus', 'pol_duration', 'pol_sit_duration', 'drv_age1', 'drv_age2',
                                    'vh_age', 'vh_cyl', 'vh_value', 'town_mean_altitude', 'population', 'vh_speed',
                                    'vh_weight']
    categorical_feature_names_sel = ['pol_coverage', 'pol_usage', 'drv_drv2', 'vh_make', 'vh_type',
                                     'vh_fuel', ]
    le = LabelEncoder()
    df2 = df[categorical_feature_names_sel].apply(le.fit_transform)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df[numerical_features_names_sel])
    num_features = np.concatenate((imp.transform(df[numerical_features_names_sel].to_numpy(dtype=np.float64)),
                                   df2.to_numpy(dtype=np.float64)),
                                  axis=1)
    preprocessor = nn_lib.Preprocessor(num_features)
    pm = PricingModel(preprocessor=preprocessor, imputer=imp, encoder=le, categorical=categorical_feature_names_sel,
                      numerical=numerical_features_names_sel)
    pm.warp(model)
    return pm


def evaluate_model():
    df = pd.read_csv('part3_data.csv').sample(frac=1)
    split_index = int(0.8*df.shape[0])
    print(df.head())
    df_train = df.iloc[:split_index, :]
    df_test = df.iloc[split_index:, :]
    x_train = df_train[df_train.columns[:-2]].to_numpy()
    y_train = df_train[df_train.columns[-1]].to_numpy()
    print(y_train)
    claim_train = df_train[df_train.columns[-2]].to_numpy()
    print(x_train)
    print(y_train)
    print(claim_train)
    x_test = df_test[df_test.columns[:-2]].to_numpy()
    y_test = df_test[df_test.columns[-1]].to_numpy()
    claim_test = df_test[df_test.columns[-2]].to_numpy()
    numerical_features_names_sel = ['pol_bonus', 'pol_duration', 'pol_sit_duration', 'drv_age1', 'drv_age2',
                                    'vh_age', 'vh_cyl', 'vh_value',  'town_mean_altitude', 'population', 'vh_speed',
                                    'vh_weight']
    categorical_feature_names_sel = ['pol_coverage', 'pol_usage', 'drv_drv2', 'vh_make', 'vh_type',
                                     'vh_fuel', ]
    le = LabelEncoder()
    df2 = df[categorical_feature_names_sel].apply(le.fit_transform)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df[numerical_features_names_sel])
    num_features = np.concatenate((imp.transform(df[numerical_features_names_sel].to_numpy(dtype=np.float64)),
                                   df2.to_numpy(dtype=np.float64)),
                                  axis=1)
    preprocessor = nn_lib.Preprocessor(num_features)
    pm = PricingModel(preprocessor=preprocessor, imputer=imp, encoder=le, categorical=categorical_feature_names_sel,
                      numerical=numerical_features_names_sel)
    print(y_test.max(), claim_test.max())
    pm.fit(x_train, y_train, claim_train)
    predicted_y = pm.predict_claim_probability(x_test)
    print(predicted_y)
    print("AUC score is %.3f" % pm.evaluate_architecture(x_test, y_test, predicted_y))
    print('Premioum', pm.predict_premium(x_test))


def train_model():
    df = pd.read_csv('part3_data.csv').sample(frac=1)
    x_train = df[df.columns[:-2]].to_numpy()
    y_train = df[df.columns[-1]].to_numpy()
    claim_train = df[df.columns[-2]].to_numpy()
    numerical_features_names_sel = ['pol_bonus', 'pol_duration', 'pol_sit_duration', 'drv_age1', 'drv_age2',
                                    'vh_age', 'vh_cyl', 'vh_value', 'town_mean_altitude', 'population', 'vh_speed',
                                    'vh_weight']
    categorical_feature_names_sel = ['pol_coverage', 'pol_usage', 'drv_drv2', 'vh_make', 'vh_type',
                                     'vh_fuel', ]
    le = LabelEncoder()
    df2 = df[categorical_feature_names_sel].apply(le.fit_transform)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(df[numerical_features_names_sel])
    num_features = np.concatenate((imp.transform(df[numerical_features_names_sel].to_numpy(dtype=np.float64)),
                                   df2.to_numpy(dtype=np.float64)),
                                  axis=1)
    preprocessor = nn_lib.Preprocessor(num_features)
    pm = PricingModel(preprocessor=preprocessor, imputer=imp, encoder=le, categorical=categorical_feature_names_sel,
                      numerical=numerical_features_names_sel)
    pm.fit(x_train, y_train, claim_train)
    pm.save_model()


# evaluate_model()
# train_model()


