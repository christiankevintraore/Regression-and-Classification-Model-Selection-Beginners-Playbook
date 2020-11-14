"""common_model_selection.py
~~~~~~~~~~~~~~

A common model selection class that will be extends by generic regressor and classifier.
It's also implements commons util methods used by regressors and classifiers ; and their main processes.

"""

#### Libraries
import numpy as np
import utils.dataset_manager as dm
from texttable import Texttable
import os



#### Main CommonModelSelection class
class CommonModelSelection:

    def __init__(self, datasetManager:dm.DatasetManager):
        """Initialize the current regressor or classifier object with the dataset manager.

        """
        self.datasetManager = datasetManager

    def evaluate(self):
        """Trains the regressor or classifier and calculate the R2 score (for regressors) or the  of Confusion Matrix and Accuracy Score (for classifiers).
        Defines the 'evaluate' method that each regressor or classifier will have to implement.
        Returns the regressor or classifier name to display and the R2 score or an array of Confusion Matrix and Accuracy Score.

        """
        raise NotImplementedError

    def predict(self):
        """Predicts dependent variables, from users command line input independent variables table.
        Defines the 'predict' method that each regressor or classifier will have to implement.
        Returns the regressor or classifier name to display and the two dimensions table of predicted dependent variables.

        """
        raise NotImplementedError

    def predictions_relevance(self):
        """Returns a comparison table for predicted values et real values of test set.
        Defines the 'predictions_relevance' method that each regressor or classifier will have to implement.
        Returns the regressor or classifier name to display and the two dimensions table of the test set with predicted values.

        """
        raise NotImplementedError

    def predict_user_input_variables(self, predictLambda):
        """Predict from a particular regressor or classifier through the input lambda.
        Returns a two dimensional table with each row containing the user input independent variables and a predicted dependent one.

        """
        userInputVariables = self.get_user_input_for_prediction()
        return self.reshape_y_pred_and_concatenate(userInputVariables, predictLambda(userInputVariables))

    def get_user_input_for_prediction(self):
        """Checks and returns the user predefined independent variables table for prediction.

        """
        valuesToPredict = self.datasetManager.params.predict
        if valuesToPredict is None:
            raise AttributeError

        X_train = self.datasetManager.X_train
        if X_train is None:
            raise LookupError("The model was not yet be evaluated at the moment")

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



def instantiate_models(modelsDictionary, datasetManager):
    models = {}
    for modelDefinition in modelsDictionary.items():
        models[modelDefinition[0]] = modelDefinition[1](datasetManager)
    return models



def evaluate_and_sort_models(models, sortLambda):
    # Training the models
    evaluations = []
    for model in models:
        evaluations.append(model.evaluate())

    # Descendent sorting of evaluations by predefined function
    evaluations.sort(key=sortLambda, reverse=True)
    return evaluations



def predict_from_lambda(models, predictLambda):
    predictions = []
    for model in models:
        predictions.append(predictLambda(model))
    return predictions



def print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='', additionalHeaders=None):
    # Printing predictions
    for prediction in predictions:
        print("\n{0}".format(prediction[0]))
        predictionTexttable = get_text_table(len(prediction[1][0]))
        predictionTexttable.set_max_width(get_console_window_width())
        predictionTexttable.add_rows(datasetManager.insert_header_to_top(str_all_bidimensional_list_elements(prediction[1]), appendToDependentVariablesHeader, additionalHeaders))
        print(predictionTexttable.draw())



def str_all_bidimensional_list_elements(oneDimensionalList):
    return [[str(element) for element in elements] for elements in oneDimensionalList]



def get_dictionary_codes(dictionary):
    return list(dictionary.keys())



def get_models_for_prediction(modelName, modelsDictionary, codesOfDesiredModels):
    if codesOfDesiredModels is None:
        return modelsDictionary.values()

    allModelsCode = get_dictionary_codes(modelsDictionary)
    models = []
    for modelCode in codesOfDesiredModels:
        if modelCode not in allModelsCode:
            raise AttributeError("Invalid {0} code for prediction : '{1}', expected codes {2}".format(modelName, modelCode, allModelsCode))

        models.append(modelsDictionary[modelCode])

    return models



def get_text_table(nbColumn):
    result = Texttable()
    result.set_max_width(get_console_window_width())
    result.set_cols_align(['c'] * nbColumn)
    result.set_cols_valign(['m'] * nbColumn)
    result.set_cols_dtype(['t'] * nbColumn)
    return result



def get_console_window_width():
    rows, columns = os.popen('stty size', 'r').read().split()
    return int(columns)
