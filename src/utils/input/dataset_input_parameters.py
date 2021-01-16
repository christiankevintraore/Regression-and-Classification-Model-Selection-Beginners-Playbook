"""dataset_input_parameters.py
~~~~~~~~~~~~~~

Wrap users command line input parameters to read a dataset.

Desirable features :
    - Make this class immutable.

"""

#### Libraries
from argparse import ArgumentParser



# Constants
DEFAULT_COLUMNS_INTERVAL_SEPARATOR = '-'



class DatasetInputParameters:

    def __init__(self, datasetFilePath, columnsIntervalSeparator=DEFAULT_COLUMNS_INTERVAL_SEPARATOR):
        """Initialize dataset input parameters values.

        """
        self.datasetFilePath = datasetFilePath
        self.columnsIntervalSeparator = columnsIntervalSeparator



def get_arguments():
    """Return the parser arguments, a list of dictionaries (representing parameters of 'add_argument' method of the ArgumentParser object), related to DatasetInputParameters object

    """
    return [{ 'name_or_flags' : 'dataset', 'type' : str, 'help' : 'A dataset file path' },
            { 'name_or_flags' : '-columnsIntervalSeparator', 'type' : str, 'default' : DEFAULT_COLUMNS_INTERVAL_SEPARATOR,\
                'help' : "Defines columns interval (inbound and outbound intervals are inclusive) separator (default: '{0}')".format(DEFAULT_COLUMNS_INTERVAL_SEPARATOR) }]



def get_dataset_input_parameters():
    """Return the dataset input parameters from predifiened object or command line entries

    """

    # Uncomment one of the commented lines below if you want to use a predefined Input Parameters object
    # return  DatasetInputParameters('src/data/Data-For-Regression.csv')
    # return  DatasetInputParameters('src/data/Data-For-Classification.csv')

    # Getting parameters from command line
    from utils.input.input_parameters_common_utils import add_arguments
    argumentParser = ArgumentParser(description='Get the input parameters to read a dataset')
    add_arguments(argumentParser, get_arguments())
    args = argumentParser.parse_args()

    return DatasetInputParameters(args.dataset, columnsIntervalSeparator=args.columnsIntervalSeparator)



def get_action(defaultValue):
    """Returns 'store_false' if the input value is True, otherwise returns 'store_true'

    """
    return 'store_false' if defaultValue else 'store_true'
