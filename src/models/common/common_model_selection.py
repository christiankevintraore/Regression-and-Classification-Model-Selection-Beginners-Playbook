"""common_model_selection.py
~~~~~~~~~~~~~~

A common model selection class that will be extends by generic regressor and classifier.
It's also implements common util methods used by regressors and classifiers ; and their main processes.

"""

#### Libraries
from pandas import DataFrame
from dataset.dataset import Dataset
from numpy import array, ndarray
from dataset.model_selection_dataset_manager import ModelSelectionDatasetManager
from utils.common_utils import reshape_second_pred_and_concatenate



#### Main CommonModelSelection class
class CommonModelSelection:

    def __init__(self, datasetManager:ModelSelectionDatasetManager):
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
        userInputVariables = self.datasetManager.params.get_user_input_for_prediction(self.datasetManager.independentVariablesColumnsNumber)
        return reshape_second_pred_and_concatenate(userInputVariables,\
            self.handle_encoded_y_pred(predictLambda(self.encode_user_input_variables(userInputVariables))))

    def encode_user_input_variables(self, userInputVariables):
        """Encodes user input variables for prediction.

        """
        datasetManager = self.datasetManager
        independentVariablesDataFrame = self.datasetManager.X
        independentVariablesToEncodeDataFrame = independentVariablesDataFrame.copy().append(DataFrame(userInputVariables,\
            columns=independentVariablesDataFrame.columns), ignore_index=True)
        dataToPredict = datasetManager.categoricalDataPreprocessor.handle_data(Dataset(datasetManager.params)\
            .load_data(independentVariablesToEncodeDataFrame)).data
        return dataToPredict[len(independentVariablesDataFrame.index):len(dataToPredict.index)]

    def get_predictions_relevance(self, X_test, y_test, y_pred):
        """Returns a truncated comparison table.

        """
        return reshape_second_pred_and_concatenate(reshape_second_pred_and_concatenate(X_test, y_test),\
                self.handle_encoded_y_pred(get_new_truncated_list(y_pred, self.datasetManager.params.nbPredictionLinesToShow)))

    def handle_encoded_y_pred(self, y_pred):
        """Decodes the input y_pred if encoded, otherwise returns the input data.

        """
        if self.datasetManager.is_y_categorical_column:
            labelsByEncodedNumber = self.datasetManager.y_labels_by_encoded_number
            encodedNumbers = labelsByEncodedNumber.keys()
            return array([labelsByEncodedNumber[element] if element in encodedNumbers else element for element in y_pred])

        return y_pred



def get_new_truncated_list(inputList, nbFirstElementsToKeep):
    """Truncates and returns a new list containing the 'nbFirstElementsToKeep' elements.

    """
    return inputList[:nbFirstElementsToKeep].copy()



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
    from utils.common_utils import get_text_table, get_console_window_width
    # Printing predictions
    for prediction in predictions:
        print("\n{0}".format(prediction[0]))
        predictionTexttable = get_text_table(len(prediction[1][0]))
        predictionTexttable.set_max_width(get_console_window_width())
        predictionTexttable.add_rows(datasetManager.insert_header_to_top(str_all_bidimensional_list_elements(prediction[1]), appendToDependentVariablesHeader, additionalHeaders))
        print(predictionTexttable.draw(), "\n", sep='')



def str_all_bidimensional_list_elements(bidimensionalList):
    return [[str(element) for element in elements] for elements in bidimensionalList]



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



def build_model_codes_with_descriptions(modelCodesList, modelDescriptionByCodes):
    """Returns a list of the input imputation strategies with descriptions.

    """
    result = []
    for modelCode in modelCodesList:
        result.append(modelCode + ' (' + modelDescriptionByCodes[modelCode] + ')')

    return result
