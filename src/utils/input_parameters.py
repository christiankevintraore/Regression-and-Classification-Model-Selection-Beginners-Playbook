"""input_parameters.py
~~~~~~~~~~~~~~

Wrap all users command line input parameters.

Desirable features :
    - Make this class immutable.

"""

#### Libraries
import argparse

# Constants
DEFAULT_NO_HEADER = False
DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX = -1
DEFAULT_INDEPENDENT_VARIABLES_START_INDEX = None
DEFAULT_INDEPENDENT_VARIABLES_END_INDEX = -1
DEFAULT_SPLIT_TEST_SIZE = 0.2
DEFAULT_SPLIT_RANDOM_STATE = 0
DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES = False
DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION = None
DEFAULT_REGRESSION_TYPES_LIST_FOR_PREDICTION = None
DEFAULT_REGRESSION_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY = None
DEFAULT_PREDICTIONS_LINES_TO_DISPLAY = 10



class InputParameters:

    def __init__(self, datasetFilePath, noHeader=DEFAULT_NO_HEADER, dependentVariableColumnIndex=DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX,\
        independentVariablesStartIndex=DEFAULT_INDEPENDENT_VARIABLES_START_INDEX, independentVariablesEndIndex=DEFAULT_INDEPENDENT_VARIABLES_END_INDEX,\
        splitTestSize=DEFAULT_SPLIT_TEST_SIZE, splitRandomState=DEFAULT_SPLIT_RANDOM_STATE, featureScaleDependentVariables=DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES,\
        predict=DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION, predictOnly=DEFAULT_REGRESSION_TYPES_LIST_FOR_PREDICTION,\
        showPredictionsFor=DEFAULT_REGRESSION_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY, nbPredictionLinesToShow=DEFAULT_PREDICTIONS_LINES_TO_DISPLAY):
        """Initialize input parameters values.

        """
        self.datasetFilePath = datasetFilePath
        self.noHeader = noHeader
        self.dependentVariableColumnIndex = dependentVariableColumnIndex
        self.independentVariablesStartIndex = independentVariablesStartIndex
        self.independentVariablesEndIndex = independentVariablesEndIndex
        self.splitTestSize = splitTestSize
        self.splitRandomState = splitRandomState
        self.featureScaleDependentVariables = featureScaleDependentVariables
        self.predict = predict
        self.predictOnly = predictOnly
        self.showPredictionsFor = showPredictionsFor
        self.nbPredictionLinesToShow = nbPredictionLinesToShow



def get_input_parameters(regressorsCode):
    """...Return the input parameters from predifiened object or command line entries

    """

    # Uncomment the line below if you want to use a predefined Input Parameters object
    # return  InputParameters('src/data/Data-For-Regression.csv')

    # Getting parameters from command line
    argumentParser = argparse.ArgumentParser(description='Determine the best regressor type to use for a specific dataset.')
    argumentParser.add_argument('dataset', type=str, help='A dataset file path')
    argumentParser.add_argument('-noHeader', action=get_action(DEFAULT_NO_HEADER), help="Indicates that there is no header in the dataset (default: {0})".format(DEFAULT_NO_HEADER))
    argumentParser.add_argument('-dependentVariableColumnIndex', type=int, default=DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX, help="Indicates the unique dependent variable column index (default: {0})".format(DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX))
    argumentParser.add_argument('-independentVariablesStartIndex', type=int, default=DEFAULT_INDEPENDENT_VARIABLES_START_INDEX, help="Indicates independent variables start column index (default: {0})".format(DEFAULT_INDEPENDENT_VARIABLES_START_INDEX))
    argumentParser.add_argument('-independentVariablesEndIndex', type=int, default=DEFAULT_INDEPENDENT_VARIABLES_END_INDEX, help="Indicates independent variables end column index (default: {0})".format(DEFAULT_INDEPENDENT_VARIABLES_END_INDEX))
    argumentParser.add_argument('-splitTestSize', type=float, default=DEFAULT_SPLIT_TEST_SIZE, help="Indicates the proportion of the dataset to include in the test sets (default: {0})".format(DEFAULT_SPLIT_TEST_SIZE))
    argumentParser.add_argument('-splitRandomState', type=int, default=DEFAULT_SPLIT_RANDOM_STATE, help="Controls the shuffling applied to the data before applying the split (default: {0})".format(DEFAULT_SPLIT_RANDOM_STATE))
    argumentParser.add_argument('-featureScaleDependentVariables', action=get_action(DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES),\
        help="Indicates if the unique dependent variable should be feature scaled, if needed for the regressor (default: {0})".format(DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES))
    argumentParser.add_argument('-predict', type=float, nargs='+', action=Store_as_array, default=DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION, help="Defines independent variables for prediction (default: {0})".format(DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION))
    argumentParser.add_argument('-predictOnly', type=str, nargs='+', default=DEFAULT_REGRESSION_TYPES_LIST_FOR_PREDICTION,\
        help="Defines a list of regressors for prediction (default: {0}, means use all existing regressors). Here is the complete list of regressors codes {1}".format(DEFAULT_REGRESSION_TYPES_LIST_FOR_PREDICTION, regressorsCode))
    argumentParser.add_argument('-showPredictionsFor', type=str, nargs='+', default=DEFAULT_REGRESSION_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY,\
        help="Define a list of regressors to display a comparison table of test and predictions values sets (default: {0}, means don't show any comparison table). Here is the complete list of regressors codes {1}".format(DEFAULT_REGRESSION_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY, regressorsCode))
    argumentParser.add_argument('-nbPredictionLinesToShow', type=int, default=DEFAULT_PREDICTIONS_LINES_TO_DISPLAY,\
        help="Indicates the number of lines to display for the comparison table (default: {0}), only applicable with -nbPredictionLinesToShow parameter".format(DEFAULT_PREDICTIONS_LINES_TO_DISPLAY))
    
    args = argumentParser.parse_args()

    return InputParameters(args.dataset, noHeader=args.noHeader, dependentVariableColumnIndex=args.dependentVariableColumnIndex,\
            independentVariablesStartIndex=args.independentVariablesStartIndex, independentVariablesEndIndex=args.independentVariablesEndIndex,\
            splitTestSize=args.splitTestSize, splitRandomState=args.splitRandomState, featureScaleDependentVariables=args.featureScaleDependentVariables,\
            predict=args.predict, predictOnly=args.predictOnly, showPredictionsFor=args.showPredictionsFor, nbPredictionLinesToShow=args.nbPredictionLinesToShow)



def get_action(defaultValue):
    return 'store_false' if defaultValue else 'store_true'



class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if namespace.predict is None:
            namespace.predict = []

        namespace.predict.append(values)
