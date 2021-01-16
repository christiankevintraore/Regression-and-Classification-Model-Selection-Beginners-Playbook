"""dataset_manager.py
~~~~~~~~~~~~~~~~~~~~~~~

Runs multiple classification models on the same dataset on purpose to determine the best accuracy score.

Desirable features :
    - Add unit tests.

"""

#### Libraries
from utils.input.dataset_manager_input_parameters import get_input_parameters, DatasetManagerInputParameters
from dataset.dataset import Dataset
from utils.common_utils import print_horizontal_rule, fill_up_empty_table_data, print_execution_time
from numpy import reshape
from pandas import isnull
from utils.input.input_parameters_common_utils import ColumnsAndOperationsParser



def main():
    """Dataset info main process.

    """
    from sys import path
    path.append("../src/")

    # Global variables declaration
    selectedColumns = None
    selectedDataset = None

    # Get input parameters for dataset management
    inputParameters:DatasetManagerInputParameters = get_input_parameters()

    # Load the data
    dataset = Dataset(inputParameters).load_data()

    # Print the dataset info
    if not inputParameters.hideInfo:
        print_horizontal_rule('Initial dataset')
        print_info(dataset)
    
    # Select columns from dataset
    if inputParameters.select is not None:
        # Instantiating the new dataset
        columnsAndOperationsParser = ColumnsAndOperationsParser(inputParameters.columnsIntervalSeparator)
        selectedColumns = columnsAndOperationsParser.get_all_selected_columns(dataset, inputParameters.select)
        selectedDataset = Dataset(inputParameters).load_data(dataset.data.iloc[:, selectedColumns].copy())

        # Applying operations on the new dataset
        apply_operations(selectedDataset, inputParameters)

        # Printing the new dataset
        print_horizontal_rule('Selected dataset')
        print_info(selectedDataset)
    
    # Save the new dataset
    newDatasetPath = inputParameters.saveTo
    if (newDatasetPath is not None) and (selectedDataset.data is not None):
        selectedDataset.data.to_csv(newDatasetPath)
        print_horizontal_rule("New dataset saved to : {0}".format(newDatasetPath))



def apply_operations(selectedDataset, inputParameters:DatasetManagerInputParameters):
    """Applies input operations on defined columns of the input dataset.

    """
    data = selectedDataset.data
    allDatasetHeaders = selectedDataset.get_all_headers()
    operationsToApply = inputParameters.apply
    columnsAndOperationsParser = ColumnsAndOperationsParser(inputParameters.columnsIntervalSeparator)
    if operationsToApply is not None:
        for definedOperationsAndColumns in operationsToApply:
            columns, operations = columnsAndOperationsParser.get_columns_and_operations(selectedDataset, definedOperationsAndColumns)
            for column in columns:
                columnName = allDatasetHeaders[column]
                for operation in operations:
                    data[columnName] = data[columnName].apply(lambda x: eval(operation, {'x' : x, 'isnull' : isnull}))



def print_info(dataset):
    from utils.common_utils import get_text_table
    # Printing a sample data
    print(dataset.data.head(), "\n")

    # Printing the dataset info
    print(dataset.data.info())

    # Printing the null values columns table
    nullValuesDictionary = dataset.count_columns_null_values(dataset.get_all_headers())
    nullValuesListToDisplay = []
    for columnName in nullValuesDictionary.keys():
            nullValuesListToDisplay.append([columnName, nullValuesDictionary[columnName]])
    
    nullValuesListToDisplay.sort(key=lambda columnNameWithNbNullValues: columnNameWithNbNullValues[1], reverse=True)
    nullValuesTexttable = get_text_table(2)
    fill_up_empty_table_data(nullValuesListToDisplay, 2)
    nullValuesListToDisplay.insert(0, ["Column Name", "Number of Null values"])
    nullValuesTexttable.add_rows(nullValuesListToDisplay)
    print("\n", nullValuesTexttable.draw(), sep='')

    # Printing the null values columns table
    categoricalColumnsName = dataset.get_categorical_columns_name()
    categoricalColumnsName = reshape(categoricalColumnsName, (len(categoricalColumnsName), 1)).tolist()
    categoricalColumnsTexttable = get_text_table(1)
    fill_up_empty_table_data(categoricalColumnsName, 1)
    categoricalColumnsName.insert(0, ["Categorical Features columns name"])
    categoricalColumnsTexttable.add_rows(categoricalColumnsName)
    print("\n", categoricalColumnsTexttable.draw(), "\n", sep='')



if __name__ == "__main__":
    from timeit import timeit
    print_execution_time(timeit("main()", setup="from __main__ import main", number=1))
