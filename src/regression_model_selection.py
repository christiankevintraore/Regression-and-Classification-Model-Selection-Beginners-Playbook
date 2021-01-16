"""regression_model_selection.py
~~~~~~~~~~~~~~~~~~~~~~~

Runs multiple regression models on the same dataset on purpose to determine the best R2 score.

Desirable features :
    - Add more regressors.
    - Add unit tests.
    - Do something for Simple Linear Regression.

"""

#### Libraries
from dataset.model_selection_dataset_manager import ModelSelectionDatasetManager
from utils.input.model_selection_input_parameters import ModelSelectionInputParameters
from utils.input.data_preprocessor_input_parameters import DataPreprocessorInputParameters
from models.common.common_model_selection import build_model_codes_with_descriptions, get_dictionary_codes, instantiate_models,\
    evaluate_and_sort_models, predict_from_lambda, get_models_for_prediction, print_predictions
from utils.common_utils import print_execution_time
# Regressors
from models.regressors.multiple_linear_regression import MultipleLinearRegressor as MLRReg
from models.regressors.polynomial_regression import PolynomialRegressor as PolyReg
from models.regressors.support_vector_regression import SupportVectorRegressor as SVRReg
from models.regressors.decision_tree_regression import DecisionTreeRegressor as DTRReg
from models.regressors.random_forest_regression import RandomForestRegressor as RFRReg



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

# Dictionary of descriptions of all existing regressors.
REGRESSORS_DESCRIPTIONS = {
    'MLR' : 'Multiple Linear Regressor',
    'POLY' : 'Polynomial Regressor',
    'SVR' : 'Support Vector Regressor',
    'DTR' : 'Decision Tree Regressor',
    'RFR' : 'Random Forest Regressor'
}

def main():
    """Regression main process.

    """
    from sys import path
    path.append("../src/")

    # Get input parameters for dataset management
    from utils.input.model_selection_input_parameters import get_input_parameters as get_model_selection_input_parameters, get_arguments_namespace
    from utils.input.data_preprocessor_input_parameters import get_input_parameters as get_data_preprocessor_input_parameters, get_input_parameters_arguments
    inputArguments = get_arguments_namespace('regressor', build_model_codes_with_descriptions(get_dictionary_codes(EXISTING_REGRESSORS), REGRESSORS_DESCRIPTIONS),\
        get_input_parameters_arguments())
    modelSelectionInputParameters:ModelSelectionInputParameters = get_model_selection_input_parameters(inputArguments)
    dataPreprocessorInputParameters:DataPreprocessorInputParameters = get_data_preprocessor_input_parameters(inputArguments)

    # Load the data
    datasetManager = ModelSelectionDatasetManager(modelSelectionInputParameters, dataPreprocessorInputParameters)
    datasetManager.load_and_split_data()

    # Instantiate the regressors dictionary
    regressorsDictionary = instantiate_models(EXISTING_REGRESSORS, datasetManager)

    # Get the evaluations
    evaluations = evaluate_and_sort_models(regressorsDictionary.values(), get_R2_score)
    
    # Print evaluations
    print_evaluations(evaluations.copy())

    modelName = 'regressor'
    # Predict from users input
    if modelSelectionInputParameters.predict is not None:
        predictLambda = lambda regressor : regressor.predict()
        predictions = predict_from_lambda(get_models_for_prediction(modelName, regressorsDictionary, modelSelectionInputParameters.predictOnly), predictLambda)
        print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='Predicted ')

    # Display predictions comparison
    if modelSelectionInputParameters.showPredictionsFor is not None:
        predictLambda = lambda regressor : regressor.predictions_relevance()
        predictions = predict_from_lambda(get_models_for_prediction(modelName, regressorsDictionary, modelSelectionInputParameters.showPredictionsFor), predictLambda)
        print_predictions(datasetManager, predictions, additionalHeaders=[ "Predicted {0}".format(datasetManager.get_dependent_variable_header()) ])



def print_evaluations(evaluations):
    from utils.common_utils import get_text_table
    # Printing evaluations
    for evaluation in evaluations:
        evaluation[1] = "{:.20f}".format(evaluation[1])

    predictionTexttable = get_text_table(2)
    evaluations.insert(0, ["Regression Model", "R2 Score"])
    predictionTexttable.add_rows(evaluations)
    print(predictionTexttable.draw(), "\n", sep='')



def get_R2_score(evaluation):
    return evaluation[1]



if __name__ == "__main__":
    from timeit import timeit
    print_execution_time(timeit("main()", setup="from __main__ import main", number=1))
