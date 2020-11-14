"""classification_model_selection.py
~~~~~~~~~~~~~~~~~~~~~~~

Runs multiple classification models on the same dataset on purpose to determine the best accuracy score.

Desirable features :
    - Add more classifier.
    - Add unit tests.

"""

#### Libraries
import utils.dataset_manager as dm
import utils.input_parameters as ip
from texttable import Texttable
import models.common.common_model_selection as cms
# Classifiers
from models.classifiers.decision_tree_classification import DecisionTreeClassifier as DTClassif
from models.classifiers.k_nearest_neighbors_classification import KNearestNeighborsClassifier as KNNClassif
from models.classifiers.kernel_svm_classification import KernelSvmClassifier as KSVMClassif
from models.classifiers.logistic_regression_classification import LogisticRegressionClassifier as LRClassif
from models.classifiers.naive_bayes_classification import NaiveBayesClassifier as NBClassif
from models.classifiers.random_forest_classification import RandomForestClassifier as RFClassif
from models.classifiers.support_vector_machine_classification import SupportVectorMachineClassifier as SVMClassif

import sys
sys.path.append("../src/")



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

def main():
    """Classification main process.

    """
    # Get input parameters for dataset management
    inputParameters = ip.get_input_parameters('classifier', cms.get_dictionary_codes(EXISTING_CLASSIFIERS))

    # Load the data
    datasetManager = dm.DatasetManager(inputParameters)
    datasetManager.load_and_split_data()

    # Instantiate the classifiers dictionary
    classifiersDictionary = cms.instantiate_models(EXISTING_CLASSIFIERS, datasetManager)

    # Get the evaluations
    evaluations = cms.evaluate_and_sort_models(classifiersDictionary.values(), get_accuracy_score)
    
    # Print evaluations
    print_evaluations(evaluations)

    modelName = 'classifier'
    # Predict from users input
    if inputParameters.predict is not None:
        predictLambda = lambda classifier : classifier.predict()
        predictions = cms.predict_from_lambda(cms.get_models_for_prediction(modelName, classifiersDictionary, inputParameters.predictOnly), predictLambda)
        cms.print_predictions(datasetManager, predictions, appendToDependentVariablesHeader='Predicted ')

    # Display predictions comparison
    if inputParameters.showPredictionsFor is not None:
        predictLambda = lambda classifier : classifier.predictions_relevance()
        predictions = cms.predict_from_lambda(cms.get_models_for_prediction(modelName, classifiersDictionary, inputParameters.showPredictionsFor), predictLambda)
        cms.print_predictions(datasetManager, predictions, additionalHeaders=[ "Predicted {0}".format(datasetManager.get_dependent_variable_header()) ])



def print_evaluations(evaluations):
    # Printing evaluations
    evaluationsToPrint = []
    for evaluation in evaluations:
        evaluationsToPrint.append([evaluation[0], "{:.20f}".format(evaluation[2]), str(evaluation[1][0][0]), str(evaluation[1][0][1]), str(evaluation[1][1][1]), str(evaluation[1][1][0]), str(evaluation[1][0][0] + evaluation[1][1][1]), str(evaluation[1][0][1] + evaluation[1][1][0])])

    predictionTexttable = cms.get_text_table(8)
    predictionTexttable.set_max_width(cms.get_console_window_width())
    evaluationsToPrint.insert(0, ["Classification Model", "Accuracy Score", "Number of True Positives", "Number of False Positives", "Number of True Negatives", "Number of False Negatives", "Number of True Predictions", "Number of False Predictions"])
    predictionTexttable.add_rows(evaluationsToPrint)
    print(predictionTexttable.draw())



def get_accuracy_score(evaluation):
    return evaluation[2]



if __name__ == "__main__":
    main()
