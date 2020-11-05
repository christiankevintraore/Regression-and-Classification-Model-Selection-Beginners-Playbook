"""regression_model_selection.py
~~~~~~~~~~~~~~~~~~~~~~~

Runs multiple regression models on the same dataset on purpose to determine the best R2 score.

Desirable features :
    - Add more regressors.
    - Add unit tests.
    - Do something for Simple Linear Regression.

"""

#### Libraries
import utils.dataset_manager as dm
import utils.input_parameters as ip
from texttable import Texttable
# Regressors
from models.regressors.multiple_linear_regression import MultipleLinearRegressor as MLRReg
from models.regressors.polynomial_regression import PolynomialRegressor as PolyReg
from models.regressors.support_vector_regression import SupportVectorRegressor as SVRReg
from models.regressors.decision_tree_regression import DecisionTreeRegressor as DTRReg
from models.regressors.random_forest_regression import RandomForestRegressor as RFRReg

import sys
sys.path.append("../src/")



# Dictionary of all existing regressors.
# For each row, the key is a regressor code (that can be used for restricting a specific prediction). And the value is the regressor class
# To avoid processing a specific regressor, feel free to comment the concerned line
EXISTING_REGRESSORS = {
    'MLR' : MLRReg,
    'POLY' : PolyReg,
    'SVR' : SVRReg,
    'DTR' : DTRReg,
    'RFR' : RFRReg
}

def main():
    """...

    """
    # Get input parameters for dataset management
    inputParameters = ip.get_input_parameters(get_existing_regressors_code())

    # Load the data
    datasetManager = dm.DatasetManager(inputParameters)
    datasetManager.load_and_split_data()

    # Instantiate the regressors dictionary
    regressorsDictionary = get_regressors(datasetManager)

    # Get the evaluations
    evaluations = evaluate_and_sort_regressors(regressorsDictionary.values())
    
    # Print evaluations
    print_evaluations(evaluations.copy())

    # Predict from users input
    if inputParameters.predict is not None:
        predictLambda = lambda regressor : regressor.predict()
        predictions = predict_from_lambda(get_regressors_for_prediction(regressorsDictionary, inputParameters.predictOnly), predictLambda)
        print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='Predicted ')

    # Display predictions comparison
    if inputParameters.showPredictionsFor is not None:
        predictLambda = lambda regressor : regressor.predictions_relevance()
        predictions = predict_from_lambda(get_regressors_for_prediction(regressorsDictionary, inputParameters.showPredictionsFor), predictLambda)
        print_predictions(datasetManager, predictions, additionalHeaders=[ "Predicted {0}".format(datasetManager.get_dependent_variable_header()) ])



def get_regressors(datasetManager):
    # To avoid processing a specific regressor, feel free to comment the concerned line
    regressors = {}
    for regressorDefinition in EXISTING_REGRESSORS.items():
        regressors[regressorDefinition[0]] = regressorDefinition[1](datasetManager)
    return regressors



def evaluate_and_sort_regressors(regressors):
    # Training the regressors
    evaluations = []
    for regressor in regressors:
        evaluations.append(regressor.evaluate())

    # Descendent sorting of evaluations by R2 score
    evaluations.sort(key=get_R2_score, reverse=True)
    return evaluations



def predict_from_lambda(regressors, predictLambda):
    predictions = []
    for regressor in regressors:
        predictions.append(predictLambda(regressor))
    return predictions



def print_evaluations(evaluations):
    # Printing evaluations
    for evaluation in evaluations:
        evaluation[1] = "{:.20f}".format(evaluation[1])

    predictionTexttable = Texttable()
    predictionTexttable.set_cols_dtype(['t', 't'])
    evaluations.insert(0, ["Regression Type", "R2 Score"])
    predictionTexttable.add_rows(evaluations)
    print(predictionTexttable.draw())



def print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='', additionalHeaders=None):
    # Printing predictions
    for prediction in predictions:
        print("\n      {0}".format(prediction[0]))
        predictionTexttable = Texttable()
        predictionTexttable.add_rows(datasetManager.insert_header_to_top(prediction[1], appendToDependentVariablesHeader, additionalHeaders))
        print(predictionTexttable.draw())



def get_R2_score(evaluation):
    return evaluation[1]



def get_existing_regressors_code():
    return list(EXISTING_REGRESSORS.keys())



def get_regressors_for_prediction(regressorsDictionary, codesOfDesiredRegressors):
    if codesOfDesiredRegressors is None:
        return regressorsDictionary.values()

    allRegressorsCode = list(regressorsDictionary.keys())
    regressors = []
    for regressorCode in codesOfDesiredRegressors:
        if regressorCode not in allRegressorsCode:
            raise AttributeError("Invalid regressor code for prediction : '{0}', expected codes {1}".format(regressorCode, allRegressorsCode))

        regressors.append(regressorsDictionary[regressorCode])

    return regressors



if __name__ == "__main__":
    main()
