"""model_selection_dataset_manager.py
~~~~~~~~~~~~~~

An implementation of generic dataset manager for loading and splitting the dataset into a training and a test sets, based on inputs variables.
Depending of the regressor type, this class can be also used for feature scaling, still based on initialization variables.

Desirable features :
    - Read big dataset (chunksize -> TextParser).
    - More flexibility for independent and dependent variables definition.
    - Auto détection if should do feature scaling.
    - Handle missing data.
    - Handle categorical data.

"""

#### Libraries
from numpy import reshape
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset.dataset import Dataset
from utils.input.model_selection_input_parameters import ModelSelectionInputParameters
from utils.input.data_preprocessor_input_parameters import DataPreprocessorInputParameters
from utils.input.input_parameters_common_utils import ColumnsAndOperationsParser
from dataset.preprocessors.utils.categorical_data_preprocessor import CategoricalDataPreprocessor



#### Main ModelSelectionDatasetManager class
class ModelSelectionDatasetManager:

    #### Private variables, to avoid (makes it a little bit less easy) changes from external
    __dataset = None

    def __init__(self, params:ModelSelectionInputParameters, dataPreprocessorParams:DataPreprocessorInputParameters):
        """Initialize the current ModelSelectionDatasetManager object with required variables.

        """
        self._ModelSelectionDatasetManager__dataset = Dataset(params)
        self.params = params
        self.categoricalDataPreprocessor:CategoricalDataPreprocessor = CategoricalDataPreprocessor(dataPreprocessorParams)
        self.dataPreprocessorParams = dataPreprocessorParams
        self.dependentVariableHeader = None
        self.independentVariablesHeaders = None
        self.headersToDisplay = None
        self.independentVariablesColumnsIndex = None
        self.dependentVariableColumnIndex = None

    def load_data(self):
        """Initialize independent variables tables into X variable, and the dependent one into Y.

        """
        from dataset.preprocessors.utils.missing_data_preprocessor import MissingDataPreprocessor
        dataset = self._ModelSelectionDatasetManager__dataset.load_data()

        self.independentVariablesColumnsIndex = self.get_independent_variables_columns_index()
        self.independentVariablesColumnsNumber = len(self.independentVariablesColumnsIndex)
        self.dependentVariableColumnIndex = self.get_dependent_variable_column_index()
        if self.dependentVariableColumnIndex in self.independentVariablesColumnsIndex:
            raise AttributeError("Dependent variable column cannot be part of independent variables columns")

        #Data preprocessing
        self.independentVariablesDataset = Dataset(self.params).load_data(dataset.data.iloc[:, self.independentVariablesColumnsIndex].copy())
        MissingDataPreprocessor(self.dataPreprocessorParams).handle_data(self.independentVariablesDataset, self.independentVariablesColumnsIndex)

        self.X = self.independentVariablesDataset.data

        y_column_name = self.get_dependent_variable_header()
        self.y = dataset.data.iloc[:, self.dependentVariableColumnIndex].to_frame(name=y_column_name)
        self.is_y_categorical_column = len(Dataset(self.params).load_data(self.y).get_categorical_columns_name()) > 0
        self.y_labels_by_encoded_number = dict(enumerate(self.y[y_column_name].astype('category').cat.categories))\
            if self.is_y_categorical_column else None

    def split_data(self):
        """Split independent variables "X" and dependent "y" into training sets (X_train & y_train) and test sets (X_test & y_test).

        """
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_test_for_predictions_relevance,\
            self.y_test_for_predictions_relevance = encode_X_and_y_training_and_test_sets(self.X, self.y, self.params,\
                self.categoricalDataPreprocessor, self.is_y_categorical_column, self.get_number_of_prediction_lines_to_show())
        
        self.categoricalDataPreprocessor.print_accumulated_encoding_results()

    def load_and_split_data(self):
        """Load and split the dataset, from one call.

        """
        self.load_data()
        self.split_data()

    def get_dependent_variable_column_index(self):
        """Returns the unique dependent variable column index.

        """
        columnsAndOperationsParser = ColumnsAndOperationsParser(self.params.columnsIntervalSeparator)
        return columnsAndOperationsParser.get_all_selected_columns(self._ModelSelectionDatasetManager__dataset, [[ self.params.dependentVariableColumn ]], False)[0]

    def get_independent_variables_columns_index(self):
        """Returns all independent variables columns indexes.

        """
        independentVariablesColumns = self.params.independentVariablesColumns
        dataset = self._ModelSelectionDatasetManager__dataset
        if independentVariablesColumns is None:
            return list(range(dataset.get_number_of_columns() - 1))

        columnsAndOperationsParser = ColumnsAndOperationsParser(self.params.columnsIntervalSeparator)
        return columnsAndOperationsParser.get_all_selected_columns(dataset, independentVariablesColumns, False)
    
    def insert_header_to_top(self, elements, appendToDependentVariablesHeader='', additionalHeaders=None):
        headers = self.get_independent_variables_header() + [self.get_dependent_variable_header(appendToDependentVariablesHeader)]

        if additionalHeaders is not None:
            headers = headers + additionalHeaders

        elements.insert(0, headers)
        return elements

    def get_independent_variables_header(self):
        if(self.independentVariablesHeaders is not None):
            return self.independentVariablesHeaders

        allHeaders = self._ModelSelectionDatasetManager__dataset.get_all_headers()
        independentVariablesHeaders = [allHeaders[index] for index in self.independentVariablesColumnsIndex]

        self.independentVariablesHeaders = independentVariablesHeaders
        return independentVariablesHeaders

    def get_dependent_variable_header(self, appendToDependentVariablesHeader=''):
        if(self.dependentVariableHeader is not None):
            return appendToDependentVariablesHeader + self.dependentVariableHeader

        allHeaders = self._ModelSelectionDatasetManager__dataset.get_all_headers()
        dependentVariableHeader = allHeaders[self.dependentVariableColumnIndex]

        self.dependentVariableHeader = dependentVariableHeader
        return appendToDependentVariablesHeader + dependentVariableHeader

    def get_number_of_samples(self):
        return len(self._ModelSelectionDatasetManager__dataset.data.index)

    def get_number_of_prediction_lines_to_show(self):
        """Returns the 'nbPredictionLinesToShow' from input parameters, if there is some predictions to show. Otherwise returns None.

        """
        if self.params.showPredictionsFor is not None:
            return self.params.nbPredictionLinesToShow

        return None



def encode_X_and_y_training_and_test_sets(X:DataFrame, y:DataFrame, modelSelectionInputParameters:ModelSelectionInputParameters,\
    categoricalDataPreprocessor:CategoricalDataPreprocessor, is_y_categorical_column:bool, nbRowsForOriginalTestSets=None):
    """Encodes dependant (y) and independant (X) variables training sets (X_train & y_train) and test sets (X_test & y_test).

    """
    from dataset.preprocessors.utils.categorical_data_preprocessor import apply_label_encoding
    from models.common.common_model_selection import get_new_truncated_list

    encoded_X = categoricalDataPreprocessor.handle_data(Dataset(modelSelectionInputParameters).load_data(X.copy())).data
    encoded_y = apply_label_encoding(y.copy()) if is_y_categorical_column else y

    X_train, X_test, y_train, y_test = train_test_split(encoded_X, encoded_y, test_size = modelSelectionInputParameters.splitTestSize,\
            random_state = modelSelectionInputParameters.splitRandomState)
    
    original_X_test = None
    original_y_test = None
    if nbRowsForOriginalTestSets is not None:
        original_X_test = get_new_truncated_list(X.iloc[list(X_test.index)], nbRowsForOriginalTestSets)
        original_y_test = get_new_truncated_list(y.iloc[list(y_test.index)], nbRowsForOriginalTestSets)

    return X_train, X_test, y_train, y_test, original_X_test, original_y_test



def reshape_y_set_split_data(datasetManager:ModelSelectionDatasetManager):
    """Reshape the dependent variables set (y, from one dimension to two) and split the dataset into training sets (X_train & y_train) and test sets (X_test & y_test).

    """
    y = datasetManager.y
    y = DataFrame(reshape(y, (len(y), 1)), index=y.index, columns=y.columns)

    X_train, X_test, y_train, y_test, datasetManager.X_test_for_predictions_relevance, datasetManager.y_test_for_predictions_relevance\
        = encode_X_and_y_training_and_test_sets(datasetManager.X, y, datasetManager.params, datasetManager.categoricalDataPreprocessor,\
            datasetManager.is_y_categorical_column, datasetManager.get_number_of_prediction_lines_to_show())

    return X_train, X_test, y_train, y_test



def do_feature_scaling(inputTable:DataFrame):
    """Do feature scaling on the input table. Necessary for certain algorithm like Support Vector Regression.
    Returns the scaler and the scaled table.

    """
    scaler = StandardScaler()
    return scaler, DataFrame(scaler.fit_transform(inputTable), index=inputTable.index, columns=inputTable.columns)
