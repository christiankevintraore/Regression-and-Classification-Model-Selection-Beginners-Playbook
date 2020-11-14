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
import models.common.common_model_selection as cms
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
    """Regression main process.

    """
    # Get input parameters for dataset management
    inputParameters = ip.get_input_parameters('regressor', cms.get_dictionary_codes(EXISTING_REGRESSORS))

    # Load the data
    datasetManager = dm.DatasetManager(inputParameters)
    datasetManager.load_and_split_data()

    # Instantiate the regressors dictionary
    regressorsDictionary = cms.instantiate_models(EXISTING_REGRESSORS, datasetManager)

    # Get the evaluations
    evaluations = cms.evaluate_and_sort_models(regressorsDictionary.values(), get_R2_score)
    
    # Print evaluations
    print_evaluations(evaluations.copy())

    modelName = 'regressor'
    # Predict from users input
    if inputParameters.predict is not None:
        predictLambda = lambda regressor : regressor.predict()
        predictions = cms.predict_from_lambda(cms.get_models_for_prediction(modelName, regressorsDictionary, inputParameters.predictOnly), predictLambda)
        cms.print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='Predicted ')

    # Display predictions comparison
    if inputParameters.showPredictionsFor is not None:
        predictLambda = lambda regressor : regressor.predictions_relevance()
        predictions = cms.predict_from_lambda(cms.get_models_for_prediction(modelName, regressorsDictionary, inputParameters.showPredictionsFor), predictLambda)
        cms.print_predictions(datasetManager, predictions, additionalHeaders=[ "Predicted {0}".format(datasetManager.get_dependent_variable_header()) ])



def print_evaluations(evaluations):
    # Printing evaluations
    for evaluation in evaluations:
        evaluation[1] = "{:.20f}".format(evaluation[1])

    predictionTexttable = cms.get_text_table(2)
    evaluations.insert(0, ["Regression Model", "R2 Score"])
    predictionTexttable.add_rows(evaluations)
    print(predictionTexttable.draw())



def get_R2_score(evaluation):
    return evaluation[1]



if __name__ == "__main__":
    main()
