"""common_utils.py
~~~~~~~~~~~~~~

A collection of common util methods used by main processes.

"""

#### Libraries
from numpy import concatenate, reshape
from os import popen
from texttable import Texttable




# Constants
DEFAULT_NOT_APPLICABLE_SYMBOL = 'N / A'
DEFAULT_OPERATION_LEADING_SYMBOL = '->'
OPERATIONS_DEFINITION_EXAMPLES = "e.g. : '1 2 3 4 {0} x if isnull(x) else x.strip()', or '1-4 {0} x / 2 {0} + 42', or 'col1 col2 {0} 99',  or 'col0-col2 {0} x * 5', where x correspond to a cell value.".format(DEFAULT_OPERATION_LEADING_SYMBOL)



def reshape_second_pred_and_concatenate(firstTable, secondTable):
    """Reshapes the secondTable table (from one dimension to two) and concatenates it with the 'firstTable' table.

    """
    return concatenate((firstTable, reshape(secondTable, (len(secondTable), 1))) ,1)



def get_text_table(nbColumn):
    result = Texttable()
    result.set_max_width(get_console_window_width())
    result.set_cols_align(['c'] * nbColumn)
    result.set_cols_valign(['m'] * nbColumn)
    result.set_cols_dtype(['t'] * nbColumn)
    return result



def fill_up_empty_table_data(tableData, nbColumn):
    if len(tableData) == 0:
        tableData.append([DEFAULT_NOT_APPLICABLE_SYMBOL] * nbColumn)



def get_console_window_width():
    rows, columns = popen('stty size', 'r').read().split()
    return int(columns)



def print_execution_time(executionTime):
    from datetime import timedelta
    print("\nProcessed in {0}\n".format(str(timedelta(seconds=executionTime))))



def is_int(inputValue):
    #TODO Replace by a regex check
    try:
        int(inputValue)
        return True
    except:
        return False



def is_float(inputValue):
    #TODO Replace by a regex check
    try:
        float(inputValue)
        return True
    except:
        return False



def to_upper(inputValue):
    if inputValue is not None:
        return inputValue.upper()
    
    return inputValue



def print_horizontal_rule(title=None):
    width = get_console_window_width()
    hr = None

    if(title is None):
        hr = '\n+' + ('=' * (width - 2)) + '+\n'

    else:
        semiHr = '+' + ('=' * int((width - 6 - len(title)) / 2)) + '+'
        hr = semiHr + ' ' + title + ' ' + semiHr + '\n'
    
    print(hr)



def flatten(listOfElementsList:list):
    try:
        return [element for sublist in listOfElementsList for element in sublist]
    except:
        return listOfElementsList



def is_url(filePath):
    from validators import url
    return url(filePath)
