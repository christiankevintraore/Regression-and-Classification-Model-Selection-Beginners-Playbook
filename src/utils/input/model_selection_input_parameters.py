"""model_selection_input_parameters.py
~~~~~~~~~~~~~~

Wrap all users command line input parameters for model selection.

Desirable features :
    - Make this class immutable.

"""

#### Libraries
from argparse import ArgumentParser, _StoreAction
from utils.input.dataset_input_parameters import DatasetInputParameters, DEFAULT_COLUMNS_INTERVAL_SEPARATOR, get_arguments, get_action
from utils.input.input_parameters_common_utils import add_arguments, append_values
from utils.common_utils import is_float, is_int

# Constants
DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX = '-1'
DEFAULT_INDEPENDENT_VARIABLES_COLUMNS = None
DEFAULT_SPLIT_TEST_SIZE = 0.2
DEFAULT_SPLIT_RANDOM_STATE = 0
DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES = False
DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION = None
DEFAULT_MODEL_TYPES_LIST_FOR_PREDICTION = None
DEFAULT_MODEL_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY = None
DEFAULT_PREDICTIONS_LINES_TO_DISPLAY = 10



class ModelSelectionInputParameters(DatasetInputParameters):

    def __init__(self, datasetFilePath, columnsIntervalSeparator=DEFAULT_COLUMNS_INTERVAL_SEPARATOR, dependentVariableColumn=DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX,\
        independentVariablesColumns=DEFAULT_INDEPENDENT_VARIABLES_COLUMNS, splitTestSize=DEFAULT_SPLIT_TEST_SIZE, splitRandomState=DEFAULT_SPLIT_RANDOM_STATE,\
        featureScaleDependentVariables=DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES, predict=DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION,\
        predictOnly=DEFAULT_MODEL_TYPES_LIST_FOR_PREDICTION, showPredictionsFor=DEFAULT_MODEL_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY,\
        nbPredictionLinesToShow=DEFAULT_PREDICTIONS_LINES_TO_DISPLAY):
        """Initialize input parameters values.

        """
        super().__init__(datasetFilePath, columnsIntervalSeparator)

        self.dependentVariableColumn = dependentVariableColumn
        self.independentVariablesColumns = independentVariablesColumns
        self.splitTestSize = splitTestSize
        self.splitRandomState = splitRandomState
        self.featureScaleDependentVariables = featureScaleDependentVariables
        self.predict = predict
        self.predictOnly = predictOnly
        self.showPredictionsFor = showPredictionsFor
        self.nbPredictionLinesToShow = nbPredictionLinesToShow
        self.userInputsForPrediction = None

    def get_user_input_for_prediction(self, independentVariablesColumnsNumber):
        """Checks and returns the user predefined independent variables table for prediction.

        """
        if self.userInputsForPrediction is not None:
            return self.userInputsForPrediction.copy()

        valuesToPredict = self.predict
        if valuesToPredict is None:
            raise AttributeError

        for oneSetToPredict in valuesToPredict:
            oneSetToPredictLen = len(oneSetToPredict)
            if independentVariablesColumnsNumber != oneSetToPredictLen:
                raise AttributeError("Invalid independent variables set to predict : {0}, expected {1} elements"\
                    .format(oneSetToPredict, independentVariablesColumnsNumber))

            for i in range(oneSetToPredictLen):
                currentEntry = oneSetToPredict[i]
                if is_int(currentEntry):
                    oneSetToPredict[i] = int(currentEntry)
                elif is_float(currentEntry):
                    oneSetToPredict[i] = float(currentEntry)
                else:
                    oneSetToPredict[i] = str(currentEntry)

        self.userInputsForPrediction = valuesToPredict
        return valuesToPredict



def get_arguments_namespace(modelType, modelCodesWithDescriptions, extraArguments=None):
    """Return the input parameters from predifiened object or command line entries

    """

    # Uncomment one of the commented lines below if you want to use a predefined Input Parameters object
    # return  ModelSelectionInputParameters('src/data/Data-For-Regression.csv')
    # return  ModelSelectionInputParameters('src/data/Data-For-Classification.csv')

    # Getting parameters from command line
    argumentParser = ArgumentParser(description="Determine the best {0} type to use for the specified dataset.".format(modelType))

    add_arguments(argumentParser, get_arguments())
    if extraArguments is not None:
        add_arguments(argumentParser, extraArguments)

    argumentParser.add_argument('-dependentVariableColumn', type=str, default=DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX, help="Indicates the unique dependent variables column (default: {0}, means the last one)".format(DEFAULT_DEPENDENT_VARIABLE_COLUMN_INDEX)\
                + " E.g. : -dependentVariableColumn 2, -dependentVariableColumn -3, -dependentVariableColumn col2")
    argumentParser.add_argument('-independentVariablesColumns', type=str, nargs='+', action=Store_independentVariablesColumns_as_array, default=DEFAULT_INDEPENDENT_VARIABLES_COLUMNS,\
        help="Defines a list of independent variables columns by name, or index, or interval (inbound and outbound intervals are inclusive). Default: {0}, means all columns except the last one.".format(DEFAULT_INDEPENDENT_VARIABLES_COLUMNS)\
                + " E.g. : -independentVariablesColumns 0 1 2, -independentVariablesColumns 0-2, -independentVariablesColumns col0 col1 col2, -independentVariablesColumns col0-col2")
    argumentParser.add_argument('-splitTestSize', type=float, default=DEFAULT_SPLIT_TEST_SIZE, help="Indicates the proportion of the dataset to include in the test sets (default: {0})".format(DEFAULT_SPLIT_TEST_SIZE))
    argumentParser.add_argument('-splitRandomState', type=int, default=DEFAULT_SPLIT_RANDOM_STATE, help="Controls the shuffling applied to the data before applying the split (default: {0})".format(DEFAULT_SPLIT_RANDOM_STATE))
    argumentParser.add_argument('-featureScaleDependentVariables', action=get_action(DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES),\
        help="Indicates if the unique dependent variable should be feature scaled, if needed for the {0} (default: {1})".format(modelType, DEFAULT_FEATURE_SCALE_DEPENDENT_VARIABLES))
    argumentParser.add_argument('-predict', type=str, nargs='+', action=Store_predict_as_array, default=DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION, help="Defines independent variables for prediction (default: {0})".format(DEFAULT_INDEPENDENT_VARIABLES_TABLE_FOR_PREDICTION))
    argumentParser.add_argument('-predictOnly', type=str, nargs='+', default=DEFAULT_MODEL_TYPES_LIST_FOR_PREDICTION,\
        help="Defines a list of {0}s for prediction (default: {1}, means use all existing {0}s). Here is the complete list of {0}s codes {2}".
                format(modelType, DEFAULT_MODEL_TYPES_LIST_FOR_PREDICTION, ', '.join([str(model) for model in modelCodesWithDescriptions])))
    argumentParser.add_argument('-showPredictionsFor', type=str, nargs='+', default=DEFAULT_MODEL_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY,\
        help="Define a list of {0}s to display a comparison table of test and predictions values sets (default: {1}, means don't show any comparison table). Here is the complete list of {0}s codes {2}"
                .format(modelType, DEFAULT_MODEL_TYPES_LIST_FOR_TEST_AND_PREDICTIONS_SETS_DISPLAY, ', '.join([str(model) for model in modelCodesWithDescriptions])))
    argumentParser.add_argument('-nbPredictionLinesToShow', type=int, default=DEFAULT_PREDICTIONS_LINES_TO_DISPLAY,\
        help="Indicates the number of lines to display for the comparison table (default: {0}), only applicable with -nbPredictionLinesToShow parameter".format(DEFAULT_PREDICTIONS_LINES_TO_DISPLAY))
    
    return argumentParser.parse_args()



def get_input_parameters(args):
    """Return the input parameters from a predefined object or from the input arguments namespace

    """
    
    # Uncomment one of the commented lines below if you want to use a predefined Input Parameters object
    # return  ModelSelectionInputParameters('src/data/Data-For-Regression.csv')
    # return  ModelSelectionInputParameters('src/data/Data-For-Classification.csv')

    return ModelSelectionInputParameters(args.dataset, args.columnsIntervalSeparator, dependentVariableColumn=args.dependentVariableColumn,\
            independentVariablesColumns=args.independentVariablesColumns, splitTestSize=args.splitTestSize, splitRandomState=args.splitRandomState,\
            featureScaleDependentVariables=args.featureScaleDependentVariables, predict=args.predict, predictOnly=args.predictOnly,\
            showPredictionsFor=args.showPredictionsFor, nbPredictionLinesToShow=args.nbPredictionLinesToShow)



class Store_independentVariablesColumns_as_array(_StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        append_values(namespace, 'independentVariablesColumns', values)



class Store_predict_as_array(_StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        append_values(namespace, 'predict', values)
