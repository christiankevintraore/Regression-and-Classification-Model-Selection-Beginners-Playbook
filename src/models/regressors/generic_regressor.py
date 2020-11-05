"""generic_regressor.py
~~~~~~~~~~~~~~

A generic regressor that will be extends by each regressor.
It's also implements commons util methods used by regressors.

"""

#### Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import utils.dataset_manager as dm



#### Main GenericRegressor class
class GenericRegressor:

    def __init__(self, datasetManager:dm.DatasetManager):
        """Initialize the current regressor object with input variables.

        """
        self.datasetManager = datasetManager

    def evaluate(self):
        """Trains the regressor and calculate the R2 score.
        Defines the 'evaluate' method that each regressor will have to implement.
        Returns the regressor name to display and the R2 score.

        """
        raise NotImplementedError

    def predict(self):
        """Predicts dependent variables, from users command line input independent variables table.
        Defines the 'predict' method that each regressor will have to implement.
        Returns the regressor name to display and the two dimensions table of predicted dependent variables.

        """
        raise NotImplementedError

    def predictions_relevance(self):
        """Returns a comparison table for predicted values et real values of test set.
        Defines the 'predictions_relevance' method that each regressor will have to implement.
        Returns the regressor name to display and the two dimensions table of the test set with predicted values.

        """
        raise NotImplementedError

    def predict_user_input_variables(self, predictLambda):
        """Predict from a particular regression through the input lambda.
        Returns a two dimensional table with each row containing the user input independent variables and a predicted dependent one.

        """
        userInputVariables = self.get_user_input_for_prediction()
        return self.reshape_y_pred_and_concatenate(userInputVariables, predictLambda(userInputVariables))

    def get_r2_score(self, y_test:pd.DataFrame, y_pred:pd.DataFrame):
        """Evaluates a regressor model performance with the y_test and y_pred DataFrame inputs, and returns the 'r2' score.

        """
        return r2_score(y_test, y_pred)

    def get_user_input_for_prediction(self):
        """Checks and returns the user predefined independent variables table for prediction.

        """
        valuesToPredict = self.datasetManager.params.predict
        if valuesToPredict is None:
            raise AttributeError

        X_train = self.datasetManager.X_train
        if X_train is None:
            raise LookupError("The regressor was not yet be evaluated at the moment")

        independentVariablesNumber = len(X_train[0])
        for oneSetToPredict in valuesToPredict:
            if independentVariablesNumber != len(oneSetToPredict):
                raise AttributeError("Invalid independent variables set to predict : {0}, expected {1} elements".format(oneSetToPredict, independentVariablesNumber))
        
        return valuesToPredict

    def reshape_y_pred_and_concatenate(self, firstTable, secondTable):
        """Reshapes the secondTable table (from one dimension to two) and concatenates it with the 'firstTable' table.

        """
        return np.concatenate((firstTable, np.reshape(secondTable, (len(secondTable), 1))) ,1)

    def get_new_truncated_list(self, inputList, nbFirstElementsToKeep):
        """Truncates and returns a new list containing the 'nbFirstElementsToKeep' elements.

        """
        return inputList.copy()[:nbFirstElementsToKeep]

    def truncate_predictions_relevance(self, X_test, y_test, y_pred):
            """Returns a truncated comparison table.

            """
            nbPredictionLinesToShow = self.datasetManager.params.nbPredictionLinesToShow
            return self.reshape_y_pred_and_concatenate(self.reshape_y_pred_and_concatenate\
                (self.get_new_truncated_list(X_test, nbPredictionLinesToShow), self.get_new_truncated_list(y_test, nbPredictionLinesToShow)),\
                    self.get_new_truncated_list(y_pred, nbPredictionLinesToShow))