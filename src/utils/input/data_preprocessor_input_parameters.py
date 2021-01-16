"""data_preprocessor_input_parameters.py
~~~~~~~~~~~~~~

Wrap users command line input parameters to take care of missing and categorical data.

Desirable features :
    - Make this class immutable.

"""

#### Libraries
from utils.input.dataset_input_parameters import DatasetInputParameters, DEFAULT_COLUMNS_INTERVAL_SEPARATOR, get_action

# Constants
DEFAULT_SKIP_PREPROCESSING_DETAILS_INDICATOR = False



class DataPreprocessorInputParameters(DatasetInputParameters):

    def __init__(self, datasetFilePath, defaultNumericalImputationStrategy, numericalImputationStrategy, defaultCategoricalImputationStrategy, categoricalImputationStrategy,\
        defaultCategoricalColumnsEncoding, categoricalColumnsEncoding, columnsIntervalSeparator=DEFAULT_COLUMNS_INTERVAL_SEPARATOR,\
        skipPreprocessingDetails=DEFAULT_SKIP_PREPROCESSING_DETAILS_INDICATOR):
        """Initialize dataset input parameters values.

        """
        from utils.common_utils import to_upper
        
        super().__init__(datasetFilePath, columnsIntervalSeparator)

        self.defaultNumericalImputationStrategy = to_upper(defaultNumericalImputationStrategy)
        self.numericalImputationStrategy = numericalImputationStrategy
        self.defaultCategoricalImputationStrategy = to_upper(defaultCategoricalImputationStrategy)
        self.categoricalImputationStrategy = categoricalImputationStrategy
        self.skipPreprocessingDetails = skipPreprocessingDetails
        self.defaultCategoricalColumnsEncoding = defaultCategoricalColumnsEncoding
        self.categoricalColumnsEncoding = categoricalColumnsEncoding



def get_arguments(numericalImputationStrategyCodesWithDescription, numericalImputationStrategyDefaultCode, numericalImputationStrategyForSpecificColumnsExample,\
    categoricalImputationStrategyCodesWithDescription, categoricalImputationStrategyDefaultCode, categoricalImputationStrategyForSpecificColumnsExample,\
    categoricalColumnsEncodingCodesWithDescription, categoricalColumnsEncodingDefaultCode, categoricalColumnsEncodingStrategyForSpecificColumnsExample):
    """Return a list of arguments dictionaries (representing parameters of 'add_argument' method of the ArgumentParser object), related to DataPreprocessorInputParameters object.

    """
    return [{ 'name_or_flags' : '-defaultNumericalImputationStrategy', 'type' : str, 'default' : numericalImputationStrategyDefaultCode,\
            'help' : "Define the code of default numerical imputation strategy (default: {0}). Here is the complete list : {1}".format(numericalImputationStrategyDefaultCode,\
            ', '.join([str(element) for element in numericalImputationStrategyCodesWithDescription])) },
        { 'name_or_flags' : '-numericalImputationStrategy', 'type' : str, 'nargs' : '+', 'default' : None,\
            'help' : "Define the code of numerical imputation strategy for specific columns (default: None), e.g. :  {0}".format(numericalImputationStrategyForSpecificColumnsExample) },
        { 'name_or_flags' : '-defaultCategoricalImputationStrategy', 'type' : str, 'default' : categoricalImputationStrategyDefaultCode,\
            'help' : "Define the code of default categorical imputation strategy (default: {0}). Here is the complete list : {1}".format(categoricalImputationStrategyDefaultCode,\
            ', '.join([str(element) for element in categoricalImputationStrategyCodesWithDescription])) },
        { 'name_or_flags' : '-categoricalImputationStrategy', 'type' : str, 'nargs' : '+', 'default' : None,\
            'help' : "Define the code of categorical imputation strategy for specific columns (default: None), e.g. :  {0}".format(categoricalImputationStrategyForSpecificColumnsExample) },
        { 'name_or_flags' : '-defaultCategoricalColumnsEncoding', 'type' : str, 'default' : categoricalColumnsEncodingDefaultCode,\
            'help' : "Define the code of default categorical columns encoding (default: {0}). Here is the complete list : {1}".format(categoricalColumnsEncodingDefaultCode,\
            ', '.join([str(element) for element in categoricalColumnsEncodingCodesWithDescription])) },
        { 'name_or_flags' : '-categoricalColumnsEncoding', 'type' : str, 'nargs' : '+', 'default' : None,\
            'help' : "Define the code of categorical columns encoding strategy for specific columns (default: None), e.g. :  {0}".format(categoricalColumnsEncodingStrategyForSpecificColumnsExample) },
        { 'name_or_flags' : '-skipPreprocessingDetails', 'action' : get_action(DEFAULT_SKIP_PREPROCESSING_DETAILS_INDICATOR),\
            'help' : "Indicates if we should not display the data imputation and encoding details (default: {0}).".format(DEFAULT_SKIP_PREPROCESSING_DETAILS_INDICATOR) }]



def get_input_parameters_arguments():
    """Returns the input parameters arguments for data pre-processing.

    """
    from utils.input.data_preprocessor_input_parameters import get_arguments
    from dataset.preprocessors.utils.common_data_preprocessor import get_preprocessing_operations_codes_with_description
    from dataset.preprocessors.utils.categorical_data_preprocessor import BINARY_ENCODING_STRATEGY_CODE, EXISTING_CATEGORICAL_COLUMNS_ENCODING_DESCRIPTIONS,\
        EXISTING_CATEGORICAL_COLUMNS_ENCODING_STRATEGY, SPECIFIC_COLUMNS_ENCODING_EXAMPLE
    from dataset.preprocessors.utils.missing_data_preprocessor import CATEGORICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE,\
        EXISTING_CATEGORICAL_IMPUTATION_STRATEGY, EXISTING_IMPUTATION_STRATEGIES_DESCRIPTIONS, EXISTING_NUMERICAL_IMPUTATION_STRATEGY,\
        FREQUENT_IMPUTATION_STRATEGY_CODE, MEAN_IMPUTATION_STRATEGY_CODE, NUMERICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE
    
    return get_arguments(get_preprocessing_operations_codes_with_description(EXISTING_NUMERICAL_IMPUTATION_STRATEGY, EXISTING_IMPUTATION_STRATEGIES_DESCRIPTIONS),\
                MEAN_IMPUTATION_STRATEGY_CODE, NUMERICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE,\
                get_preprocessing_operations_codes_with_description(EXISTING_CATEGORICAL_IMPUTATION_STRATEGY, EXISTING_IMPUTATION_STRATEGIES_DESCRIPTIONS),\
                FREQUENT_IMPUTATION_STRATEGY_CODE, CATEGORICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE,\
                get_preprocessing_operations_codes_with_description(EXISTING_CATEGORICAL_COLUMNS_ENCODING_STRATEGY, EXISTING_CATEGORICAL_COLUMNS_ENCODING_DESCRIPTIONS),\
                BINARY_ENCODING_STRATEGY_CODE, SPECIFIC_COLUMNS_ENCODING_EXAMPLE)



def get_input_parameters(args):
    """Return the DataPreprocessorInputParameters object from command line entries

    """
    return DataPreprocessorInputParameters(args.dataset, args.defaultNumericalImputationStrategy, args.numericalImputationStrategy, args.defaultCategoricalImputationStrategy,\
        args.categoricalImputationStrategy, args.defaultCategoricalColumnsEncoding, args.categoricalColumnsEncoding, args.columnsIntervalSeparator, args.skipPreprocessingDetails)
