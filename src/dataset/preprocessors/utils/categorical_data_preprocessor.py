"""categorical_data_preprocessor.py
~~~~~~~~~~~~~~~~~~~~~~~

Takes care of categorical data of a dataset based on defined strategies.

Desirable features :
    - Dependent variable encoding.
    - Decode back to categorical.

"""

#### Libraries
from pandas import DataFrame
from dataset.dataset import Dataset
from pandas import get_dummies, concat
from category_encoders import BinaryEncoder, BackwardDifferenceEncoder
from utils.common_utils import fill_up_empty_table_data
from utils.input.data_preprocessor_input_parameters import DataPreprocessorInputParameters
from dataset.preprocessors.utils.common_data_preprocessor import CommonDataPreprocessor, get_preprocessing_operation_code_with_description



# Encoding strategies codes definition
LABEL_ENCODING_STRATEGY_CODE = 'LABEL'
ONE_HOT_ENCODING_STRATEGY_CODE = 'ONEHOT'
BINARY_ENCODING_STRATEGY_CODE = 'BINARY'
BACKWARD_ENCODING_STRATEGY_CODE = 'BACKWARD'



#### Main CategoricalDataPreprocessor class
class CategoricalDataPreprocessor(CommonDataPreprocessor):

    def __init__(self, dataPreprocessorInputParameters:DataPreprocessorInputParameters):
        """Initialize the current categorical data preprocessor.

        """
        super().__init__()

        self.dataPreprocessorInputParameters = dataPreprocessorInputParameters
        self.categoricalColumnsEncodingStrategyCodes = list(EXISTING_CATEGORICAL_COLUMNS_ENCODING_STRATEGY.keys())
        self.encodedColumnsByEncodingCodes = {}

    def handle_data(self, dataset:Dataset):
        """Handles categorical columns encoding and updates the dataset based on input parameters.

        """
        allDatasetHeaders = dataset.get_all_headers()
        categoricalColumnsByEncodingCodes = self.get_categorical_columns_by_encoding_codes(dataset)
        for encodingCode in categoricalColumnsByEncodingCodes.keys():
            encodingFunction = EXISTING_CATEGORICAL_COLUMNS_ENCODING_STRATEGY[encodingCode]
            categoricalColumnsName = self.column_indexes_to_names(categoricalColumnsByEncodingCodes[encodingCode], allDatasetHeaders)

            for categoricalColumnName in categoricalColumnsName:
                dataset.data = concat([dataset.data.drop(categoricalColumnName, axis='columns'),\
                    encodingFunction(dataset.data[categoricalColumnName].copy())], axis='columns')
        
        return dataset

    def get_categorical_columns_by_encoding_codes(self, dataset:Dataset):
        """Returns a dictionary of columns with missing data, with the imputation strategy code as key.

        """
        categoricalColumnsEncodingStrategyCodes = self.categoricalColumnsEncodingStrategyCodes
        dataPreprocessorInputParameters = self.dataPreprocessorInputParameters
        columnsIntervalSeparator = dataPreprocessorInputParameters.columnsIntervalSeparator
        
        defaultCategoricalColumnsEncoding = dataPreprocessorInputParameters.defaultCategoricalColumnsEncoding
        self.check_preprocessing_code(defaultCategoricalColumnsEncoding, categoricalColumnsEncodingStrategyCodes)

        categoricalColumnsSpecificEncodingDefinition = self.get_columns_by_strategy_code(dataset, columnsIntervalSeparator, dataPreprocessorInputParameters.categoricalColumnsEncoding)
        self.check_preprocessing_code_list(categoricalColumnsSpecificEncodingDefinition.keys(), categoricalColumnsEncodingStrategyCodes)
        
        self.check_for_duplicate_column_declaration([categoricalColumnsSpecificEncodingDefinition])
        
        categoricalColumnsIndexes = dataset.get_indexes(dataset.get_categorical_columns_name())
        result = {}
        for index in categoricalColumnsIndexes:
            currentEncodingStrategyCode = self.get_column_strategy_code(index, defaultCategoricalColumnsEncoding, categoricalColumnsSpecificEncodingDefinition)
            self.add_columns_and_preprocessing_strategy_code_into_dictionary(result, [index], currentEncodingStrategyCode)

        self.add_encoding_result(result, dataset.get_all_headers())

        return result

    def add_encoding_result(self, encodingResult:dict, allDatasetHeaders:list):
        """Adds the input encoding result into the results accumulator.

        """
        encodedColumnsByEncodingCodesKeys = self.encodedColumnsByEncodingCodes.keys()
        for encodingCode in encodingResult.keys():
            if encodingCode not in encodedColumnsByEncodingCodesKeys:
                self.encodedColumnsByEncodingCodes[encodingCode] = set()
            
            self.encodedColumnsByEncodingCodes[encodingCode].update(["'" + columnName + "'"\
                for columnName in self.column_indexes_to_names(encodingResult[encodingCode], allDatasetHeaders)])

    def print_accumulated_encoding_results(self):
        """Prints accumulated encoded columns with their encoding codes.

        """
        if not self.dataPreprocessorInputParameters.skipPreprocessingDetails:
            from utils.common_utils import get_text_table, get_console_window_width
            encodingTexttable = get_text_table(2)
            encodingTexttable.set_max_width(get_console_window_width())
            encodingToPrint = self.get_encoding_table_to_print()
            encodingToPrint.insert(0, ['Encoding Code applied', 'Columns updated'])
            encodingTexttable.add_rows(encodingToPrint)
            print(encodingTexttable.draw(), "\n", sep='')

    def get_encoding_table_to_print(self):
        """Returns accumulated encoded columns with their encoding codes.

        """
        result = []
        encodedColumnsByEncodingCodes = self.encodedColumnsByEncodingCodes
        for encodingCode in encodedColumnsByEncodingCodes.keys():
            result.append([get_preprocessing_operation_code_with_description(encodingCode, EXISTING_CATEGORICAL_COLUMNS_ENCODING_DESCRIPTIONS),\
                ', '.join([element for element in encodedColumnsByEncodingCodes[encodingCode]])])

        fill_up_empty_table_data(result, 2)
        return result



def apply_label_encoding(inputColumns:DataFrame):
    """Applies Label encoding on the input categorical columns.

    """
    for columnName in list(inputColumns.columns):
        inputColumns = concat([inputColumns.drop(columnName, axis='columns'),\
                            DataFrame(inputColumns[columnName].astype('category').cat.codes,\
                                columns=[columnName])], axis='columns')
    return inputColumns



def apply_one_hot_encoding(inputColumns:DataFrame):
    """Applies One-Hot encoding on the input categorical columns.

    """
    return get_dummies(inputColumns)



def apply_binary_encoding(inputColumns:DataFrame):
    """Applies Binary encoding on the input categorical columns.

    """
    return BinaryEncoder().fit_transform(inputColumns)



def apply_backward_difference_encoding(inputColumns:DataFrame):
    """Applies Backward Difference encoding on the input categorical columns.

    """
    return BackwardDifferenceEncoder().fit_transform(inputColumns)



# Dictionaries of all existing categorical columns encoding.
# For each row, the key is a encoding code, and the value is the function appling the categorical column encoding.
# To avoid using a specific encoding, feel free to comment the concerned line.
EXISTING_CATEGORICAL_COLUMNS_ENCODING_STRATEGY = {
    LABEL_ENCODING_STRATEGY_CODE : apply_label_encoding,
    ONE_HOT_ENCODING_STRATEGY_CODE : apply_one_hot_encoding,
    BINARY_ENCODING_STRATEGY_CODE : apply_binary_encoding,
    BACKWARD_ENCODING_STRATEGY_CODE : apply_backward_difference_encoding,
}

# Dictionary of descriptions of all existing categorical columns encoding.
EXISTING_CATEGORICAL_COLUMNS_ENCODING_DESCRIPTIONS = {
    LABEL_ENCODING_STRATEGY_CODE : 'Label encoding',
    ONE_HOT_ENCODING_STRATEGY_CODE : 'One-Hot encoding',
    BINARY_ENCODING_STRATEGY_CODE : 'Binary encoding',
    BACKWARD_ENCODING_STRATEGY_CODE : 'Backward Difference encoding',
}

# Specific categorical columns encoding example
SPECIFIC_COLUMNS_ENCODING_EXAMPLE = "-categoricalColumnsEncoding '0 2 -> {0}' 'col1 col2 -> {1}' '3-5 -> {2}'".format(ONE_HOT_ENCODING_STRATEGY_CODE,\
    BINARY_ENCODING_STRATEGY_CODE, BACKWARD_ENCODING_STRATEGY_CODE)
