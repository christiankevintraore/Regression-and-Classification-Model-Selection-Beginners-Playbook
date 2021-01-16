"""dataset_manager_input_parameters.py
~~~~~~~~~~~~~~

Wrap all users command line input parameters for dataset manager.

Desirable features :
    - Make this class immutable.

"""

#### Libraries
from argparse import ArgumentParser, _StoreAction
from utils.input.dataset_input_parameters import DatasetInputParameters, get_arguments, get_action
from utils.common_utils import OPERATIONS_DEFINITION_EXAMPLES
from utils.input.input_parameters_common_utils import add_arguments, append_values

# Constants
DEFAULT_HIDE_INFO = False
DEFAULT_COLUMNS_TO_SELECT = None
DEFAULT_OUTPUT_FILE_PATH = None
DEFAULT_OPERATIONS_TO_APPLY = None



class DatasetManagerInputParameters(DatasetInputParameters):

    def __init__(self, datasetFilePath, columnsIntervalSeparator, hideInfo=DEFAULT_HIDE_INFO, select=DEFAULT_COLUMNS_TO_SELECT,\
        saveTo=DEFAULT_OUTPUT_FILE_PATH, apply=DEFAULT_OPERATIONS_TO_APPLY):
        """Initialize input parameters values.

        """
        super().__init__(datasetFilePath, columnsIntervalSeparator)

        self.hideInfo = hideInfo
        self.select = select
        self.saveTo = saveTo
        self.apply = apply



def get_input_parameters():
    """Return the input parameters from predifiened object or command line entries

    """

    # Uncomment one of the commented lines below if you want to use a predefined Input Parameters object
    # return  DatasetManagerInputParameters('src/data/Data-For-Regression.csv')
    # return  DatasetManagerInputParameters('src/data/Data-For-Classification.csv')

    # Getting parameters from command line
    argumentParser = ArgumentParser(description="Show some information about the specified dataset. Can also create a new dataset from the specified one, based on defined operations in command line.")
    add_arguments(argumentParser, get_arguments())
    argumentParser.add_argument('-hideInfo', action=get_action(DEFAULT_HIDE_INFO), help="Indicates if the process should hide dataset info (default: {0})".format(DEFAULT_HIDE_INFO))
    argumentParser.add_argument('-select', type=str, nargs='+', action=Store_select_as_array, default=DEFAULT_COLUMNS_TO_SELECT,\
        help="Defines a list of columns by name, interval (inbound and outbound intervals are inclusive) or index to select for a new dataset (default: {0}, '*' means all columns)".format(DEFAULT_COLUMNS_TO_SELECT))
    argumentParser.add_argument('-saveTo', type=str, default=DEFAULT_OUTPUT_FILE_PATH,\
        help="Defines the output file path were to save the new dataset (default: {0})".format(DEFAULT_OUTPUT_FILE_PATH))
    argumentParser.add_argument('-apply', type=str, nargs='+', default=DEFAULT_OPERATIONS_TO_APPLY,\
        help="Defines a list of operations (in quotes) to be applied to on some columns of the selected dataset (default: {0}). {1}".format(DEFAULT_OPERATIONS_TO_APPLY, OPERATIONS_DEFINITION_EXAMPLES))
    
    args = argumentParser.parse_args()

    return DatasetManagerInputParameters(args.dataset, args.columnsIntervalSeparator, hideInfo=args.hideInfo, select=args.select,\
        saveTo=args.saveTo, apply=args.apply)



class Store_select_as_array(_StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        append_values(namespace, 'select', values)
