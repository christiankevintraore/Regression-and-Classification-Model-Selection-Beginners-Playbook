"""common_data_preprocessor.py
~~~~~~~~~~~~~~~~~~~~~~~

A common data preprocessor which defines common functions for data preprocessing.

Desirable features :
    - ...

"""

#### Libraries
from dataset.dataset import Dataset
from utils.common_utils import DEFAULT_NOT_APPLICABLE_SYMBOL, to_upper, fill_up_empty_table_data
from utils.input.input_parameters_common_utils import ColumnsAndOperationsParser



#### Main CommonDataPreprocessor class
class CommonDataPreprocessor():

    def __init__(self):
        """Initialize global variables of the current common data preprocessor.

        """

    def get_column_strategy_code(self, columnIndex, defaultStrategyCode, columnsByStrategyCodesDefinitions:dict):
        """Returns the related code of the input column index from definitions, otherwise returns the default one.

        """
        for strategyCode in columnsByStrategyCodesDefinitions.keys():
            if columnIndex in columnsByStrategyCodesDefinitions[strategyCode]:
                return strategyCode

        return defaultStrategyCode

    def check_for_duplicate_column_declaration(self, columnsByStrategyCodesDefinitions:list):
        """Checks for duplicate columns declaration in definitionss.

        """
        allDeclaredColumns = []
        for columnsByStrategyCodesDefinition in columnsByStrategyCodesDefinitions:
            self.check_for_duplicate_with_existing_list(allDeclaredColumns, columnsByStrategyCodesDefinition)

    def check_for_duplicate_with_existing_list(self, alreadyDeclaredColumns, columnsByStrategyCodesDefinition):
        """Checks for duplicate columns declaration in specific preprocessing definition.

        """
        declaredColumns = [item for sublist in columnsByStrategyCodesDefinition.values() for item in sublist]
        self.check_duplicate_items_in_list(declaredColumns)
        alreadyDeclaredColumns.extend(declaredColumns)
        self.check_duplicate_items_in_list(alreadyDeclaredColumns)

    def check_duplicate_items_in_list(self, inputList):
        """Checks for duplicate columns declaration in specific preprocessing definition.

        """
        items = []
        duplicates = set()
        [x for x in inputList if (x in items and duplicates.add(x)) or (x not in items and items.append(x))]

        if len(duplicates) > 0:
            raise AttributeError("Duplicated columns : {0}.".format(duplicates))

    def get_columns_by_strategy_code(self, dataset:Dataset, columnsIntervalSeparator:str, columnsAndStrategyCodes):
        """Returns a dictionary of user defined columns set, with the preprocessing strategies code as keys.

        """
        columnsAndOperationsParser = ColumnsAndOperationsParser(columnsIntervalSeparator)
        result = {}

        if columnsAndStrategyCodes is not None:
            for columnsAndStrategyCode in columnsAndStrategyCodes:
                columns, preprocessingStrategiesCodes = columnsAndOperationsParser.get_columns_and_operations(dataset, columnsAndStrategyCode, False)

                if len(preprocessingStrategiesCodes) != 1:
                    raise AttributeError("Invalid code declaration : {0}. Expected only one code.".format(columnsAndStrategyCode))

                self.add_columns_and_preprocessing_strategy_code_into_dictionary(result, columns, to_upper(preprocessingStrategiesCodes[0]))

        return result

    def add_columns_and_preprocessing_strategy_code_into_dictionary(self, dictionary:dict, columns:list, preprocessingCode):
        """Adds columns set into the input dictionary, with the preprocessing strategy code as keys.

        """
        if preprocessingCode in dictionary:
            dictionary[preprocessingCode].update(columns)
        else:
            dictionary[preprocessingCode] = set(columns)

    def check_preprocessing_code_list(self, preprocessingCodes, existingStrategiesCodes):
        """Verify if each preprocessing codes in the input list is good.

        """
        for preprocessingCode in preprocessingCodes:
            self.check_preprocessing_code(preprocessingCode, existingStrategiesCodes)

    def check_preprocessing_code(self, preprocessingCode, existingStrategiesCodes):
        """Verify if the input preprocessing code is a good one.

        """
        if preprocessingCode not in existingStrategiesCodes:
            raise AttributeError("Invalid code : {0}, expected one of these codes : {1}".format(preprocessingCode, ', '.join([str(code) for code in existingStrategiesCodes])))
    
    def column_indexes_to_names(self, columnIndexes:list, allDatasetHeaders:list):
        """Returns the column names from the input columns indexes list.

        """
        result = []
        for columnIndex in columnIndexes:
            result.append(allDatasetHeaders[columnIndex])

        return result
    
    def columns_list_to_label(self, columns:list, allDatasetHeaders:list):
        """Returns columns with their preprocessing operations codes.

        """
        result = ', '.join(["'" + allDatasetHeaders[column] + "'" for column in columns])
        return result if len(result) > 0 else DEFAULT_NOT_APPLICABLE_SYMBOL

    def split_into_numerical_and_categorical_columns(self, columns:list, categoricalColumnsIndexes:list):
        """Splits the input columns list into two numerical and categorical lists.

        """
        numericalColumns = []
        categoricalColumns = []

        for column in columns:
            if column in categoricalColumnsIndexes:
                categoricalColumns.append(column)
            else:
                numericalColumns.append(column)

        return numericalColumns, categoricalColumns



def get_preprocessing_operations_codes_with_description(imputationStrategyDictionary:dict, preprocessingOperationsWithDescriptions:dict):
    """Returns a list of the input preprocessing operation code with their description.

    """
    result = []
    for imputationCode in imputationStrategyDictionary.keys():
        result.append(get_preprocessing_operation_code_with_description(imputationCode, preprocessingOperationsWithDescriptions))

    return result

def get_preprocessing_operation_code_with_description(imputationCode:str, preprocessingOperationsWithDescriptions:dict):
    """Returns the input preprocessing operation code with his description.

    """
    return imputationCode + ' (' + preprocessingOperationsWithDescriptions[imputationCode] + ')'
