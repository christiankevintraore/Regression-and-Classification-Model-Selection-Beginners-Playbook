"""dataset-manager.py
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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils.input_parameters as ip



#### Main DatasetManager class
class DatasetManager:

    #### Private variables, to avoid (makes it a little bit less easy) changes from external
    __dataset = None

    def __init__(self, params:ip.InputParameters):
        """Initialize the current DatasetManager object with required variables.

        """
        self.params = params
        self.dependentVariableHeader = None
        self.headers = None

    def load_data(self):
        """Initialize independent variables tables into X variable, and the dependent one into Y.

        """
        self._DatasetManager__dataset = pd.read_csv(self.params.datasetFilePath, header=None) if self.params.noHeader else pd.read_csv(self.params.datasetFilePath)
        
        self.X = self._DatasetManager__dataset.iloc[:, self.params.independentVariablesStartIndex:self.params.independentVariablesEndIndex].values
        self.y = self._DatasetManager__dataset.iloc[:, self.params.dependentVariableColumnIndex].values

    def split_data(self):
        """Split independent variables "X" and dependent "y" into training sets (X_train & y_train) and test sets (X_test & y_test).

        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = self.params.splitTestSize, random_state = self.params.splitRandomState)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def load_and_split_data(self):
        """Load and split the dataset, from one call.

        """
        self.load_data()
        self.split_data()
    
    def insert_header_to_top(self, numpyArray:np.ndarray, appendToDependentVariablesHeader='', additionalHeaders=None):
        headers = self.select_dataset_header(appendToDependentVariablesHeader)
        if additionalHeaders is not None:
            headers = headers + additionalHeaders

        result = numpyArray.tolist()
        result.insert(0, headers)
        return result

    def select_dataset_header(self, appendToDependentVariablesHeader=''):
        if(self.headers is not None):
            return self.headers

        allHeaders = list(self._DatasetManager__dataset.columns)
        headers = allHeaders[self.params.independentVariablesStartIndex:self.params.independentVariablesEndIndex]
        headers.append(self.get_dependent_variable_header(appendToDependentVariablesHeader))
        self.headers = headers
        return headers

    def get_dependent_variable_header(self, appendToDependentVariablesHeader=''):
        if(self.dependentVariableHeader is not None):
            return appendToDependentVariablesHeader + self.dependentVariableHeader

        allHeaders = list(self._DatasetManager__dataset.columns)
        dependentVariableHeader = allHeaders[self.params.dependentVariableColumnIndex]
        self.dependentVariableHeader = dependentVariableHeader
        return appendToDependentVariablesHeader + dependentVariableHeader



def reshape_y_set_split_data(datasetManager:DatasetManager):
    """Reshape the dependent variables set (y, from one dimension to two) and split  the dataset into training sets (X_train & y_train) and test sets (X_test & y_test).

    """
    y = datasetManager.y
    y = y.reshape(len(y), 1)
    return train_test_split(datasetManager.X, y, test_size = datasetManager.params.splitTestSize, random_state = datasetManager.params.splitRandomState)



def do_feature_scaling(inputTable:pd.DataFrame):
    """Do feature scaling on the input table. Necessary for certain algorithm like Support Vector Regression.
    Returns the scaler and the scaled table.

    """
    scaler = StandardScaler()
    return scaler, scaler.fit_transform(inputTable)
