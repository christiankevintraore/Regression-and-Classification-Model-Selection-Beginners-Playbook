"""missing_data_preprocessor.py
~~~~~~~~~~~~~~~~~~~~~~~

Takes care of missing data of a dataset based on defined strategies.

Desirable features :
    - ...

"""

#### Libraries
from math import sqrt
from dataset.dataset import Dataset
from sklearn.impute import SimpleImputer
from pandas import DataFrame
# import datawig TODO Install for python 3.8
from impyute.imputation.cs import mice, fast_knn
from utils.common_utils import fill_up_empty_table_data
from utils.input.data_preprocessor_input_parameters import DataPreprocessorInputParameters
from dataset.preprocessors.utils.common_data_preprocessor import CommonDataPreprocessor, get_preprocessing_operation_code_with_description



# Imputation strategies codes definition
DELETE_MISSING_DATA_ROWS_IMPUTATION_STRATEGY_CODE = 'DEL'
KNN_IMPUTATION_STRATEGY_CODE = 'KNN'
MICE_IMPUTATION_STRATEGY_CODE = 'MICE'
DATAWIG_IMPUTATION_STRATEGY_CODE = 'DATAWIG'

MEAN_IMPUTATION_STRATEGY_CODE = 'MEAN'

FREQUENT_IMPUTATION_STRATEGY_CODE = 'FREQUENT'



#### Main MissingDataPreprocessor class
class MissingDataPreprocessor(CommonDataPreprocessor):

    def __init__(self, dataPreprocessorInputParameters:DataPreprocessorInputParameters):
        """Initialize the current missing data preprocessor and set the system recursion limit to 100000.

        """
        super().__init__()

        from sys import setrecursionlimit
        setrecursionlimit(100000)
        self.dataPreprocessorInputParameters = dataPreprocessorInputParameters
        self.numericalImputationStrategiesCodes = list(EXISTING_NUMERICAL_IMPUTATION_STRATEGY.keys())
        self.categoricalImputationStrategiesCodes = list(EXISTING_CATEGORICAL_IMPUTATION_STRATEGY.keys())

    def handle_data(self, dataset:Dataset, independentVariablesColumnsIndex:list):
        """Handles missing values and updates the dataset based on input parameters.

        """
        allDatasetHeaders = dataset.get_all_headers()
        numericalImputationStrategiesCodes = self.numericalImputationStrategiesCodes
        missingDataColumnsByImputationCodes = self.get_missing_data_columns_by_imputation_code(dataset, independentVariablesColumnsIndex)
        for imputationCode in missingDataColumnsByImputationCodes.keys():
            missingDataColumnsName = self.column_indexes_to_names(missingDataColumnsByImputationCodes[imputationCode], allDatasetHeaders)
            imputationFunction = None

            if imputationCode in numericalImputationStrategiesCodes:
                imputationFunction = EXISTING_NUMERICAL_IMPUTATION_STRATEGY[imputationCode]
            else:
                imputationFunction = EXISTING_CATEGORICAL_IMPUTATION_STRATEGY[imputationCode]

            self.handle_missing_data_by_imputation_code(dataset, imputationCode, imputationFunction, missingDataColumnsName)
        
        return dataset
        

    def handle_missing_data_by_imputation_code(self, dataset:Dataset, imputationCode:str, imputationFunction, missingDataColumnsName:list):
        """Applies the current imputation strategy to the input dataset.

        """
        data = dataset.data
        if imputationCode == DELETE_MISSING_DATA_ROWS_IMPUTATION_STRATEGY_CODE:
            delete_missing_data_rows(dataset, missingDataColumnsName)
        else:
            data[missingDataColumnsName] = imputationFunction(data[missingDataColumnsName].copy())

    def get_missing_data_columns_by_imputation_code(self, dataset:Dataset, independentVariablesColumnsIndex:list):
        """Returns a dictionary of columns with missing data, with the imputation strategy code as key.

        """
        numericalImputationStrategiesCodes = self.numericalImputationStrategiesCodes
        categoricalImputationStrategiesCodes = self.categoricalImputationStrategiesCodes
        dataPreprocessorInputParameters = self.dataPreprocessorInputParameters
        columnsIntervalSeparator = dataPreprocessorInputParameters.columnsIntervalSeparator
        
        defaultNumericalImputationStrategy = dataPreprocessorInputParameters.defaultNumericalImputationStrategy
        self.check_preprocessing_code(defaultNumericalImputationStrategy, numericalImputationStrategiesCodes)

        defaultCategoricalImputationStrategy = dataPreprocessorInputParameters.defaultCategoricalImputationStrategy
        self.check_preprocessing_code(defaultCategoricalImputationStrategy, categoricalImputationStrategiesCodes)

        numericalDataSpecificImputationDefinition = self.get_columns_by_strategy_code(dataset, columnsIntervalSeparator, dataPreprocessorInputParameters.numericalImputationStrategy)
        self.check_preprocessing_code_list(numericalDataSpecificImputationDefinition.keys(), numericalImputationStrategiesCodes)
        
        categoricalDataSpecificImputationDefinition = self.get_columns_by_strategy_code(dataset, columnsIntervalSeparator, dataPreprocessorInputParameters.categoricalImputationStrategy)
        self.check_preprocessing_code_list(categoricalDataSpecificImputationDefinition.keys(), categoricalImputationStrategiesCodes)
        
        self.check_for_duplicate_column_declaration([numericalDataSpecificImputationDefinition, categoricalDataSpecificImputationDefinition])
        
        categoricalColumnsIndexes = dataset.get_indexes(dataset.get_categorical_columns_name())
        columnsWithMissingDataIndexes = dataset.count_columns_null_values(independentVariablesColumnsIndex).keys()
        
        result = {}
        for index in columnsWithMissingDataIndexes:
            currentImputationStrategyCode = None

            if index in categoricalColumnsIndexes:
                currentImputationStrategyCode = self.get_column_strategy_code(index, defaultCategoricalImputationStrategy, categoricalDataSpecificImputationDefinition)
            else:
                currentImputationStrategyCode = self.get_column_strategy_code(index, defaultNumericalImputationStrategy, numericalDataSpecificImputationDefinition)
            
            self.add_columns_and_preprocessing_strategy_code_into_dictionary(result, [index], currentImputationStrategyCode)

        if not dataPreprocessorInputParameters.skipPreprocessingDetails:
            self.print_imputated_columns_with_strategy_codes(result, categoricalColumnsIndexes, dataset.get_all_headers())

        return result

    def print_imputated_columns_with_strategy_codes(self, missingDataColumnsByImputationCodes:dict, categoricalColumnsIndexes:list, allDatasetHeaders:list):
        """Prints columns with missing data with their imputation strategy codes.

        """
        from utils.common_utils import get_text_table, get_console_window_width
        imputationsTexttable = get_text_table(3)
        imputationsTexttable.set_max_width(get_console_window_width())
        imputationsToPrint = self.get_imputations_table_to_print(missingDataColumnsByImputationCodes, categoricalColumnsIndexes, allDatasetHeaders)
        imputationsToPrint.insert(0, ['Imputation Code applied', 'Numerical Columns updated', 'Categorical Columns updated'])
        imputationsTexttable.add_rows(imputationsToPrint)
        print(imputationsTexttable.draw(), "\n", sep='')

    def get_imputations_table_to_print(self, missingDataColumnsByImputationCodes:dict, categoricalColumnsIndexes:list, allDatasetHeaders:list):
        """Returns columns with missing data with their imputation strategy codes.

        """
        result = []
        
        for imputationCode in missingDataColumnsByImputationCodes.keys():
            numericalColumns, categoricalColumns = self.split_into_numerical_and_categorical_columns(missingDataColumnsByImputationCodes[imputationCode], categoricalColumnsIndexes)
            result.append([get_preprocessing_operation_code_with_description(imputationCode, EXISTING_IMPUTATION_STRATEGIES_DESCRIPTIONS),\
                self.columns_list_to_label(numericalColumns, allDatasetHeaders), self.columns_list_to_label(categoricalColumns, allDatasetHeaders)])
        
        fill_up_empty_table_data(result, 3)
        return result



def delete_missing_data_rows(dataset:Dataset, concernedColumnsNames:list):
    """Deletes missing values rows.

    """
    for columnName in concernedColumnsNames:
        dataset.data = dataset.data[dataset.data[columnName].notna()]



def apply_mean_strategy(inputColumns:DataFrame):
    """Applies mean imputation strategy on numerical missing values.

    """
    return SimpleImputer(strategy='mean').fit_transform(inputColumns)



def apply_knn_strategy(inputColumns:DataFrame):
    """Applies the K nearest neighbours algorithm to impute missing values.

    """
    return DataFrame(fast_knn(inputColumns.values, k=int(sqrt(len(inputColumns.index)))))



def apply_mice_strategy(inputColumns:DataFrame):
    """Applies the Multivariate Imputation by Chained Equation algorithm to impute missing values.

    """
    return DataFrame(mice(inputColumns.values))



def apply_datawig_strategy(inputColumns:DataFrame):
    """Applies a Machine Learning models using Deep Neural Networks algorithm (Datawig) to impute missing values.

    """
    # TODO



def apply_frequent_strategy(inputColumns:DataFrame):
    """Applies most frequent imputation strategy on categorical missing values and returns the transformed columns.

    """
    return SimpleImputer(strategy='most_frequent').fit_transform(inputColumns)



# Dictionaries of all existing imputation strategies for numerical and categorical values.
# For each row, the key is a strategy code, and the value is the function appling the imputation.
# To avoid using a specific imputation strategy, feel free to comment the concerned line.
EXISTING_NUMERICAL_IMPUTATION_STRATEGY = {
    DELETE_MISSING_DATA_ROWS_IMPUTATION_STRATEGY_CODE : delete_missing_data_rows,
    MEAN_IMPUTATION_STRATEGY_CODE : apply_mean_strategy,
    KNN_IMPUTATION_STRATEGY_CODE : apply_knn_strategy,
    MICE_IMPUTATION_STRATEGY_CODE : apply_mice_strategy,
    # DATAWIG_IMPUTATION_STRATEGY_CODE : apply_datawig_strategy #TODO
}

EXISTING_CATEGORICAL_IMPUTATION_STRATEGY = {
    DELETE_MISSING_DATA_ROWS_IMPUTATION_STRATEGY_CODE : delete_missing_data_rows,
    FREQUENT_IMPUTATION_STRATEGY_CODE : apply_frequent_strategy,
    # KNN_IMPUTATION_STRATEGY_CODE : apply_knn_strategy, #TODO
    # MICE_IMPUTATION_STRATEGY_CODE : apply_mice_strategy, #TODO
    # DATAWIG_IMPUTATION_STRATEGY_CODE : apply_datawig_strategy #TODO
}

# Dictionary of descriptions of all existing imputation strategies for numerical and categorical values.
EXISTING_IMPUTATION_STRATEGIES_DESCRIPTIONS = {
    DELETE_MISSING_DATA_ROWS_IMPUTATION_STRATEGY_CODE : 'delete missing values rows',
    MEAN_IMPUTATION_STRATEGY_CODE : 'Mean or Median imputation strategy on numerical missing values',
    FREQUENT_IMPUTATION_STRATEGY_CODE : 'Most Frequent imputation strategy on categorical missing values',
    KNN_IMPUTATION_STRATEGY_CODE : 'K nearest neighbours algorithm',
    MICE_IMPUTATION_STRATEGY_CODE : 'Multivariate Imputation by Chained Equation algorithm',
    DATAWIG_IMPUTATION_STRATEGY_CODE : 'Machine Learning models using Deep Neural Networks algorithm',
}

# Specific columns imputation strategy example
NUMERICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE = "-numericalImputationStrategy 'col1 col2 -> {0}' '3-5 -> {1}'".format(MEAN_IMPUTATION_STRATEGY_CODE, MICE_IMPUTATION_STRATEGY_CODE)
CATEGORICAL_SPECIFIC_COLUMNS_IMPUTATION_STRATEGY_EXAMPLE = "-categoricalImputationStrategy '0 2 -> {0}' 'col3-col5 -> {1}'".format(FREQUENT_IMPUTATION_STRATEGY_CODE, MICE_IMPUTATION_STRATEGY_CODE)
