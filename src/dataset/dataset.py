"""dataset.py
~~~~~~~~~~~~~~

Represents a loaded dataset.

Desirable features :
    - Read big dataset (chunksize -> TextParser).

"""

#### Libraries
from pandas import read_csv, DataFrame
from utils.input.dataset_input_parameters import DatasetInputParameters
from utils.common_utils import is_int, is_url
from utils.online_file_manager import OnlineFileManager

# Constants
NB_LINES_TO_DETERMINE_IF_HAS_HEADER = 5



#### Main Dataset class
class Dataset:

    data = None

    def __init__(self, params:DatasetInputParameters):
        """Initialize the current Dataset object with required variables.

        """
        self.params = params
        self.allHeaders = None
        datasetFilePath = self.params.datasetFilePath
        self.__isOnlineDataset = is_url(datasetFilePath)
        self.__onlineFileManager = OnlineFileManager(datasetFilePath)

    def load_data(self, existingOne:DataFrame=None):
        """Loads the dataset or assigns the existing one to the data variable.

        """
        if existingOne is not None:
            self.data = existingOne
        else:
            inputStream = self.__onlineFileManager.get_input_stream() if self.__isOnlineDataset else self.params.datasetFilePath
            self.data = read_csv(inputStream) if self.has_defined_header() else read_csv(inputStream, header=None)

            self.__isOnlineDataset = None
            self.__onlineFileManager = None
        
        return self

    def has_defined_header(self):
        from csv import Sniffer

        datasetFilePathFirstLines = self.__onlineFileManager.read_lines(NB_LINES_TO_DETERMINE_IF_HAS_HEADER)\
            if self.__isOnlineDataset else self.__read_local_file(NB_LINES_TO_DETERMINE_IF_HAS_HEADER)

        return Sniffer().has_header(''.join(datasetFilePathFirstLines))
    
    def __read_local_file(self, nbLines):
        from itertools import islice
        return list(islice(open(self.params.datasetFilePath), nbLines))

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
