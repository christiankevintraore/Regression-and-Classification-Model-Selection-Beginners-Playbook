"""dataset.py
~~~~~~~~~~~~~~

Represents a loaded dataset.

Desirable features :
    - Read big dataset (chunksize -> TextParser).

"""

#### Libraries
from pandas import read_csv, DataFrame
from utils.input.dataset_input_parameters import DatasetInputParameters
from utils.common_utils import is_int



#### Main Dataset class
class Dataset:

    data = None

    def __init__(self, params:DatasetInputParameters):
        """Initialize the current Dataset object with required variables.

        """
        self.params = params
        self.allHeaders = None

    def load_data(self, existingOne:DataFrame=None):
        """Loads the dataset or assigns the existing one to the data variable.

        """
        if existingOne is not None:
            self.data = existingOne
        else:
            self.data = read_csv(self.params.datasetFilePath) if self.has_defined_header() else read_csv(self.params.datasetFilePath, header=None)
        
        return self

    def has_defined_header(self):
        from csv import Sniffer
        from itertools import islice
        return Sniffer().has_header(''.join(islice(open(self.params.datasetFilePath), 5)))

    def get_all_headers(self):
        """Returns the dataset headers.

        """
        if(self.allHeaders is not None):
            return self.allHeaders.copy()

        allHeaders = list(self.data.columns)
        for i in range(len(allHeaders)):
            allHeaders[i] = str(allHeaders[i])

        self.allHeaders = allHeaders
        return allHeaders.copy()

    def get_number_of_columns(self):
        """Returns number of columns of the dataset.

        """
        return len(self.get_all_headers())

    def count_columns_null_values(self, columns):
        """Counts and returns a dictionary of the number of null values of the input columns list, with the column name or index as key and the null values number as value.

        """
        columnsNullValues = {}
        for column in columns:
            wholeColumn = self.data.iloc[:, int(column)] if is_int(column) else self.data[column]
            numberOfNullValues = wholeColumn.isnull().sum()
            if numberOfNullValues > 0:
                columnsNullValues[column] = numberOfNullValues
        
        return columnsNullValues

    def get_categorical_columns_name(self):
        """Returns a list of categorical columns name. We'll consider that categorical columns are those with 'object' as data type, so non numerical columns.

        """
        return list(self.data.select_dtypes(include=['object']).columns)

    def get_indexes(self, columnsName):
        """Returns a list of indexes corresponding to the input columns name list.

        """
        result = []
        for columnName in columnsName:
            result.append(self.data.columns.get_loc(columnName))

        return result
