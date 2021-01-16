"""input_parameters_common_utils.py
~~~~~~~~~~~~~~

A collection of common util methods used by input parameters processes.

"""

#### Libraries
from utils.common_utils import DEFAULT_OPERATION_LEADING_SYMBOL, OPERATIONS_DEFINITION_EXAMPLES, is_int



class ColumnsAndOperationsParser:

    def __init__(self, columnsIntervalSeparator):
        """Initialize dataset input parameters values.

        """
        self.columnsIntervalSeparator = columnsIntervalSeparator

    def get_columns_and_operations(self, selectedDataset, definedOperationsAndColumns, allColumnsAreSelectable=True):
        """Returns a tuple of selected columns and operations.

        """
        definedOperationsAndColumns = definedOperationsAndColumns.strip()
        if DEFAULT_OPERATION_LEADING_SYMBOL not in definedOperationsAndColumns:
            raise AttributeError("Invalid operation definition {0}. {1}".format(definedOperationsAndColumns, OPERATIONS_DEFINITION_EXAMPLES))
        
        operationsAndColumns = definedOperationsAndColumns.split(DEFAULT_OPERATION_LEADING_SYMBOL)
        operations = []
        i = 1
        while i < len(operationsAndColumns):
            operations.append(operationsAndColumns[i].strip())
            i += 1

        return self.get_all_selected_columns(selectedDataset, [operationsAndColumns[0].strip().split()], allColumnsAreSelectable), operations

    def get_all_selected_columns(self, dataset, selectedColumnsList, allColumnsAreSelectable=True):
        """Returns a list of indexes corresponding to the input command line columns list.

        """
        allSelectedColumnsIndexes = set()
        for selectedColumnsList in selectedColumnsList:
            allSelectedColumnsIndexes.update(self.get_selected_columns(dataset, selectedColumnsList, allColumnsAreSelectable))
        
        return sorted(allSelectedColumnsIndexes)

    def get_selected_columns(self, dataset, selectedColumnsList, allColumnsAreSelectable=True):
        """Returns a list of indexes corresponding to the input columns list.

        """
        selectedColumnsIndexes = set()
        maxIndex = dataset.get_number_of_columns() - 1
        for inputColumn in selectedColumnsList:
            if is_int(inputColumn):
                inputColumnInNumeric = self.handle_column_negative_index(dataset, int(inputColumn))
                self.verify_index(inputColumnInNumeric, maxIndex, inputColumn)
                selectedColumnsIndexes.add(inputColumnInNumeric)

            elif self.columnsIntervalSeparator in inputColumn:
                intervalsDefinitionInError = False

                splittedInterval = inputColumn.split(self.columnsIntervalSeparator)
                if len(splittedInterval) != 2:
                    intervalsDefinitionInError = True

                inboundInterval = self.handle_column_negative_index(dataset, self.get_column_index(dataset, splittedInterval[0], inputColumn))
                outboundInterval = self.handle_column_negative_index(dataset, self.get_column_index(dataset, splittedInterval[1], inputColumn))
                if (inboundInterval > outboundInterval)\
                    or self.verify_index(inboundInterval, maxIndex, inputColumn)\
                        or self.verify_index(outboundInterval, maxIndex, inputColumn):
                    intervalsDefinitionInError = True
                
                if intervalsDefinitionInError:
                    raise AttributeError(self.get_bad_index_error_message(inputColumn, self.columnsIntervalSeparator))

                if (inboundInterval == outboundInterval):
                    selectedColumnsIndexes.add(inboundInterval)

                else:
                    selectedColumnsIndexes.update(set(range(inboundInterval, outboundInterval + 1)))
            
            elif inputColumn == '*':
                if not allColumnsAreSelectable:
                    raise AttributeError("It's not allowed to select all the columns with '*'")

                selectedColumnsIndexes.update(set(range(0, dataset.get_number_of_columns())))

            else :
                selectedColumnsIndexes.add(self.get_column_index(dataset, inputColumn, inputColumn))

        return selectedColumnsIndexes

    def verify_index(self, currentIndex, maxIndex, inputColumn):
        """Checks if the input index is out of bound.

        """
        if (currentIndex < 0) or (currentIndex > maxIndex):
            raise AttributeError(self.get_bad_index_error_message(inputColumn, self.columnsIntervalSeparator))

    def get_column_index(self, dataset, column, inputColumn):
        """Returns the column index if expressed by name, otherwise returns the same 'int' index.

        """
        if is_int(column):
            return int(column)

        if column in dataset.get_all_headers():
            return dataset.get_indexes([column])[0]

        raise AttributeError(self.get_bad_index_error_message(inputColumn, self.columnsIntervalSeparator))

    def handle_column_negative_index(self, dataset, columnIndex):
        """Converts column negative index into positive, otherwise returns the same index.

        """
        if columnIndex < 0:
            return dataset.get_number_of_columns() + columnIndex

        return columnIndex

    def get_bad_index_error_message(self, inputColumn, columnsIntervalSeparator):
        """Returns the bad index error message.

        """
        return "Invalid columns interval definition : {0}, with the separator '{1}', expected one inbound and one outbound inclusive intervals".format(inputColumn, columnsIntervalSeparator)



def add_arguments(argumentParser, argumentsDictionaries):
    """Add arguments to 'ArgumentParser' object

    """
    for arguments in argumentsDictionaries.copy():
        name_or_flags = arguments['name_or_flags']
        del arguments["name_or_flags"]
        argumentParser.add_argument(name_or_flags, **arguments)



def append_values(namespace, argumentName, values):
    if getattr(namespace, argumentName) is None:
        setattr(namespace, argumentName, [])

    getattr(namespace, argumentName).append(values)
