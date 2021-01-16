"""classification_model_selection.py
~~~~~~~~~~~~~~~~~~~~~~~

Runs multiple classification models on the same dataset on purpose to determine the best accuracy score.

Desirable features :
    - Add more classifier.
    - Add unit tests.

"""

#### Libraries
from dataset.model_selection_dataset_manager import ModelSelectionDatasetManager
from utils.input.model_selection_input_parameters import ModelSelectionInputParameters
from utils.input.data_preprocessor_input_parameters import DataPreprocessorInputParameters
from models.common.common_model_selection import build_model_codes_with_descriptions, get_dictionary_codes, instantiate_models,\
    evaluate_and_sort_models, predict_from_lambda, get_models_for_prediction, print_predictions
from utils.common_utils import print_execution_time
# Classifiers
from models.classifiers.decision_tree_classification import DecisionTreeClassifier as DTClassif
from models.classifiers.k_nearest_neighbors_classification import KNearestNeighborsClassifier as KNNClassif
from models.classifiers.kernel_svm_classification import KernelSvmClassifier as KSVMClassif
from models.classifiers.logistic_regression_classification import LogisticRegressionClassifier as LRClassif
from models.classifiers.naive_bayes_classification import NaiveBayesClassifier as NBClassif
from models.classifiers.random_forest_classification import RandomForestClassifier as RFClassif
from models.classifiers.support_vector_machine_classification import SupportVectorMachineClassifier as SVMClassif



# Dictionary of all existing classifiers.
# For each row, the key is a classifier code (that can be used for restricting a specific prediction). And the value is the classifier class
# To avoid processing a specific classifier, feel free to comment the concerned line
EXISTING_CLASSIFIERS = {
    'DTC' : DTClassif,
    'KNNC' : KNNClassif,
    'KSVMC' : KSVMClassif,
    'LRC' : LRClassif,
    'NBC' : NBClassif,
    'RFC' : RFClassif,
    'SVMC' : SVMClassif
}

# Dictionary of descriptions of all existing classifiers.
CLASSIFIERS_DESCRIPTIONS = {
    'DTC' : 'Decision Tree Classifier',
    'KNNC' : 'K Nearest Neighbors Classifier',
    'KSVMC' : 'Kernel Support Vector Machine Classifier',
    'LRC' : 'Logistic Regression Classifier',
    'NBC' : 'Naive Bayes Classifier',
    'RFC' : 'Random Forest Classifier',
    'SVMC' : 'Support Vector Machine Classifier'
}

def main():
    """Classification main process.

    """
    from sys import path
    path.append("../src/")

    # Get input parameters for dataset management
    from utils.input.model_selection_input_parameters import get_input_parameters as get_model_selection_input_parameters, get_arguments_namespace
    from utils.input.data_preprocessor_input_parameters import get_input_parameters as get_data_preprocessor_input_parameters, get_input_parameters_arguments
    inputArguments = get_arguments_namespace('classifier', build_model_codes_with_descriptions(get_dictionary_codes(EXISTING_CLASSIFIERS), CLASSIFIERS_DESCRIPTIONS),\
        get_input_parameters_arguments())
    modelSelectionInputParameters:ModelSelectionInputParameters = get_model_selection_input_parameters(inputArguments)
    dataPreprocessorInputParameters:DataPreprocessorInputParameters = get_data_preprocessor_input_parameters(inputArguments)

    # Load the data
    datasetManager = ModelSelectionDatasetManager(modelSelectionInputParameters, dataPreprocessorInputParameters)
    datasetManager.load_and_split_data()

    # Instantiate the classifiers dictionary
    classifiersDictionary = instantiate_models(EXISTING_CLASSIFIERS, datasetManager)

    # Get the evaluations
    evaluations = evaluate_and_sort_models(classifiersDictionary.values(), get_accuracy_score)
    
    # Print evaluations
    print_evaluations(evaluations)

    modelName = 'classifier'
    # Predict from users input
    if modelSelectionInputParameters.predict is not None:
        predictLambda = lambda classifier : classifier.predict()
        predictions = predict_from_lambda(get_models_for_prediction(modelName, classifiersDictionary, modelSelectionInputParameters.predictOnly), predictLambda)
        print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='Predicted ')

    # Display predictions comparison
    if modelSelectionInputParameters.showPredictionsFor is not None:
        predictLambda = lambda classifier : classifier.predictions_relevance()
        predictions = predict_from_lambda(get_models_for_prediction(modelName, classifiersDictionary, modelSelectionInputParameters.showPredictionsFor), predictLambda)
        print_predictions(datasetManager, predictions, additionalHeaders=[ "Predicted {0}".format(datasetManager.get_dependent_variable_header()) ])



def print_evaluations(evaluations):
    from utils.common_utils import get_text_table, get_console_window_width
    # Printing evaluations
    evaluationsToPrint = []
    for evaluation in evaluations:
        evaluationsToPrint.append([evaluation[0], "{:.20f}".format(evaluation[2]), str(evaluation[1][0][0]), str(evaluation[1][0][1]), str(evaluation[1][1][1]), str(evaluation[1][1][0]), str(evaluation[1][0][0] + evaluation[1][1][1]), str(evaluation[1][0][1] + evaluation[1][1][0])])

    predictionTexttable = get_text_table(8)
    predictionTexttable.set_max_width(get_console_window_width())
    evaluationsToPrint.insert(0, ["Classification Model", "Accuracy Score", "Number of True Positives", "Number of False Positives", "Number of True Negatives", "Number of False Negatives", "Number of True Predictions", "Number of False Predictions"])
    predictionTexttable.add_rows(evaluationsToPrint)
    print(predictionTexttable.draw(), "\n", sep='')



def get_accuracy_score(evaluation):
    return evaluation[2]



if __name__ == "__main__":
    from timeit import timeit
    print_execution_time(timeit("main()", setup="from __main__ import main", number=1))
