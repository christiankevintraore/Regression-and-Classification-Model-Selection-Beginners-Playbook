# Regression and Classification model selection playbook

## A beginners playbook to determine the best regression or classification model applied to a dataset.

### The goal of this project is to automate Regression or Classification model selection through a simple command line.

This tool was inspired by the great course of [Machine Learning A-Z™: Hands-On Python & R In Data Science](https://udemy.com/course/machinelearning/), made by [Kirill EREMENKO](https://udemy.com/user/kirilleremenko/) and [Hadelin DE PONTEVES](https://udemy.com/user/hadelin-de-ponteves/).\
Thanks a lot, I really enjoyed it -:)\
Also notice that the datasets, for test purpose, were took from the provided resources of this course.

## Main upcoming features
* Direct data manipulation and exportation through input parameters (in few days).
* Data preprocessing : handling missing data and categorical features encoding through several predefined algorithms (in few days).
* Automatic tuning of Regressors and Classifiers input parameters to find the best accuracy scores (next big push on February or March).

## Dataset Template
* *To work properly with this tool, your dataset should have consecutive independent variable columns ; and only one dependant variable column.*
* *It is also a good idea to always have headers, it helps to identify columns on printed tables.*

## Required python modules
Depending of modules already installed in your environment ; you may have to install one or several of these modules :
* Scikit-Learn :  ```pip install -U scikit-learn```
* TextTable :     ```pip install -U texttable```
* Numpy :         ```pip install -U numpy```
* Pandas :        ```pip install -U pandas```

# Command samples
From the root directory.

## Regression

### 1. Display the helper
`python3 src/regression_model_selection.py -h`\
\
Output :
```
usage: regression_model_selection.py [-h] [-dependentVariablesColumnIndex DEPENDENTVARIABLECOLUMNINDEX] [-independentVariablesStartIndex INDEPENDENTVARIABLESSTARTINDEX]
                                     [-independentVariablesEndIndex INDEPENDENTVARIABLESENDINDEX] [-splitTestSize SPLITTESTSIZE] [-splitRandomState SPLITRANDOMSTATE] [-featureScaleDependentVariables]
                                     [-predict PREDICT [PREDICT ...]] [-predictOnly PREDICTONLY [PREDICTONLY ...]] [-showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]]
                                     [-nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW]
                                     dataset

Determine the best regressor type to use for a specific dataset.

positional arguments:
  dataset               A dataset file path

optional arguments:
  -h, --help            show this help message and exit
  -dependentVariablesColumnIndex DEPENDENTVARIABLECOLUMNINDEX
                        Indicates the unique dependent variable column index (default: -1)
  -independentVariablesStartIndex INDEPENDENTVARIABLESSTARTINDEX
                        Indicates independent variables start column index (default: None)
  -independentVariablesEndIndex INDEPENDENTVARIABLESENDINDEX
                        Indicates independent variables end column index (default: -1)
  -splitTestSize SPLITTESTSIZE
                        Indicates the proportion of the dataset to include in the test sets (default: 0.2)
  -splitRandomState SPLITRANDOMSTATE
                        Controls the shuffling applied to the data before applying the split (default: 0)
  -featureScaleDependentVariables
                        Indicates if the unique dependent variable should be feature scaled, if needed for the regressor (default: False)
  -predict PREDICT [PREDICT ...]
                        Defines independent variables for prediction (default: None)
  -predictOnly PREDICTONLY [PREDICTONLY ...]
                        Defines a list of regressors for prediction (default: None, means use all existing regressors). Here is the complete list of regressors codes ['MLR', 'POLY', 'SVR', 'DTR', 'RFR']
  -showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]
                        Define a list of regressors to display a comparison table of test and predictions values sets (default: None, means don't show any comparison table). Here is the complete list of regressors codes ['MLR', 'POLY', 'SVR', 'DTR', 'RFR']
  -nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW
                        Indicates the number of lines to display for the comparison table (default: 10), only applicable with -nbPredictionLinesToShow parameter
```

### 2. Display the R2 score table descendent sorted
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv' -featureScaleDependentVariables`\
\
Remember to always add the *-featureScaleDependentVariables* option to indicate that the dependent variables column should be feature scaled. There is not yet an auto detection for that\
It is the minimum output of this script.\
\
Output :
```
+----------------------------+------------------------+
|      Regression Model      |        R2 Score        |
+============================+========================+
|  Random Forest Regression  | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
|   Polynomial Regression    | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
|  Decision Tree Regression  | 0.92290587417794101022 |
+----------------------------+------------------------+
```

### 3. Check some predictions
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv' -featureScaleDependentVariables -showPredictionsFor MLR POLY SVR DTR RFR -nbPredictionLinesToShow 3`\
\
Output :
```
+----------------------------+------------------------+
|      Regression Model      |        R2 Score        |
+============================+========================+
|  Random Forest Regression  | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
|   Polynomial Regression    | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
|  Decision Tree Regression  | 0.92290587417794101022 |
+----------------------------+------------------------+

Multiple Linear Regression predictions comparison
+-------+-------+---------+-------+--------+--------------------+
|  AT   |   V   |   AP    |  RH   |   PE   |    Predicted PE    |
+=======+=======+=========+=======+========+====================+
| 28.66 | 77.95 | 1009.56 | 69.07 | 431.23 | 431.42761597061804 |
+-------+-------+---------+-------+--------+--------------------+
| 17.48 | 49.39 | 1021.51 | 84.53 | 460.01 | 458.56124621712934 |
+-------+-------+---------+-------+--------+--------------------+
| 14.86 | 43.14 | 1019.21 | 99.14 | 461.14 | 462.75264705004923 |
+-------+-------+---------+-------+--------+--------------------+

Polynomial Regression predictions comparison
+-------+-------+---------+-------+--------+--------------------+
|  AT   |   V   |   AP    |  RH   |   PE   |    Predicted PE    |
+=======+=======+=========+=======+========+====================+
| 28.66 | 77.95 | 1009.56 | 69.07 | 431.23 | 433.94373801558686 |
+-------+-------+---------+-------+--------+--------------------+
| 17.48 | 49.39 | 1021.51 | 84.53 | 460.01 | 457.90452605746395 |
+-------+-------+---------+-------+--------+--------------------+
| 14.86 | 43.14 | 1019.21 | 99.14 | 461.14 | 460.52469167164236 |
+-------+-------+---------+-------+--------+--------------------+

Support Vector Regression predictions comparison
+-------+-------+---------+-------+--------+--------------------+
|  AT   |   V   |   AP    |  RH   |   PE   |    Predicted PE    |
+=======+=======+=========+=======+========+====================+
| 28.66 | 77.95 | 1009.56 | 69.07 | 431.23 | 434.05242920798264 |
+-------+-------+---------+-------+--------+--------------------+
| 17.48 | 49.39 | 1021.51 | 84.53 | 460.01 |  457.938101861348  |
+-------+-------+---------+-------+--------+--------------------+
| 14.86 | 43.14 | 1019.21 | 99.14 | 461.14 |  461.031138935269  |
+-------+-------+---------+-------+--------+--------------------+

Decision Tree Regression predictions comparison
+-------+-------+---------+-------+--------+--------------+
|  AT   |   V   |   AP    |  RH   |   PE   | Predicted PE |
+=======+=======+=========+=======+========+==============+
| 28.66 | 77.95 | 1009.56 | 69.07 | 431.23 |    431.28    |
+-------+-------+---------+-------+--------+--------------+
| 17.48 | 49.39 | 1021.51 | 84.53 | 460.01 |    459.59    |
+-------+-------+---------+-------+--------+--------------+
| 14.86 | 43.14 | 1019.21 | 99.14 | 461.14 |    460.06    |
+-------+-------+---------+-------+--------+--------------+

Random Forest Regression predictions comparison
+-------+-------+---------+-------+--------+-------------------+
|  AT   |   V   |   AP    |  RH   |   PE   |   Predicted PE    |
+=======+=======+=========+=======+========+===================+
| 28.66 | 77.95 | 1009.56 | 69.07 | 431.23 |      434.049      |
+-------+-------+---------+-------+--------+-------------------+
| 17.48 | 49.39 | 1021.51 | 84.53 | 460.01 |      458.785      |
+-------+-------+---------+-------+--------+-------------------+
| 14.86 | 43.14 | 1019.21 | 99.14 | 461.14 | 463.0200000000001 |
+-------+-------+---------+-------+--------+-------------------+
```

### 4. Make some predictions
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv' -featureScaleDependentVariables -predict 23.04 59.43 1010.23 68.99 -predict 18.5 51.43 1010.82 92.04`\
\
Output :
```
+----------------------------+------------------------+
|      Regression Model      |        R2 Score        |
+============================+========================+
|  Random Forest Regression  | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
|   Polynomial Regression    | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
|  Decision Tree Regression  | 0.92290587417794101022 |
+----------------------------+------------------------+

Multiple Linear Regression predictions
+-------+-------+---------+-------+--------------------+
|  AT   |   V   |   AP    |  RH   |    Predicted PE    |
+=======+=======+=========+=======+====================+
| 23.04 | 59.43 | 1010.23 | 68.99 | 446.95203526002575 |
+-------+-------+---------+-------+--------------------+
| 18.5  | 51.43 | 1010.82 | 92.04 | 454.1962201391441  |
+-------+-------+---------+-------+--------------------+

Polynomial Regression predictions
+-------+-------+---------+-------+--------------------+
|  AT   |   V   |   AP    |  RH   |    Predicted PE    |
+=======+=======+=========+=======+====================+
| 23.04 | 59.43 | 1010.23 | 68.99 | 444.77549954321876 |
+-------+-------+---------+-------+--------------------+
| 18.5  | 51.43 | 1010.82 | 92.04 |  453.532156674235  |
+-------+-------+---------+-------+--------------------+

Support Vector Regression predictions
+-------+-------+---------+-------+--------------------+
|  AT   |   V   |   AP    |  RH   |    Predicted PE    |
+=======+=======+=========+=======+====================+
| 23.04 | 59.43 | 1010.23 | 68.99 | 445.19682179584964 |
+-------+-------+---------+-------+--------------------+
| 18.5  | 51.43 | 1010.82 | 92.04 | 452.82985047737026 |
+-------+-------+---------+-------+--------------------+

Decision Tree Regression predictions
+-------+-------+---------+-------+--------------+
|  AT   |   V   |   AP    |  RH   | Predicted PE |
+=======+=======+=========+=======+==============+
| 23.04 | 59.43 | 1010.23 | 68.99 |    442.99    |
+-------+-------+---------+-------+--------------+
| 18.5  | 51.43 | 1010.82 | 92.04 |    459.42    |
+-------+-------+---------+-------+--------------+

Random Forest Regression predictions
+-------+-------+---------+-------+-------------------+
|  AT   |   V   |   AP    |  RH   |   Predicted PE    |
+=======+=======+=========+=======+===================+
| 23.04 | 59.43 | 1010.23 | 68.99 | 442.9899999999999 |
+-------+-------+---------+-------+-------------------+
| 18.5  | 51.43 | 1010.82 | 92.04 |      456.378      |
+-------+-------+---------+-------+-------------------+
```

## Classification

### 1. Display the helper
`python3 src/classification_model_selection.py -h`\
\
Output :
```
usage: classification_model_selection.py [-h] [-dependentVariablesColumnIndex DEPENDENTVARIABLECOLUMNINDEX] [-independentVariablesStartIndex INDEPENDENTVARIABLESSTARTINDEX]
                                         [-independentVariablesEndIndex INDEPENDENTVARIABLESENDINDEX] [-splitTestSize SPLITTESTSIZE] [-splitRandomState SPLITRANDOMSTATE] [-featureScaleDependentVariables]
                                         [-predict PREDICT [PREDICT ...]] [-predictOnly PREDICTONLY [PREDICTONLY ...]] [-showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]]
                                         [-nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW]
                                         dataset

Determine the best classifier type to use for a specific dataset.

positional arguments:
  dataset               A dataset file path

optional arguments:
  -h, --help            show this help message and exit
  -dependentVariablesColumnIndex DEPENDENTVARIABLECOLUMNINDEX
                        Indicates the unique dependent variable column index (default: -1)
  -independentVariablesStartIndex INDEPENDENTVARIABLESSTARTINDEX
                        Indicates independent variables start column index (default: None)
  -independentVariablesEndIndex INDEPENDENTVARIABLESENDINDEX
                        Indicates independent variables end column index (default: -1)
  -splitTestSize SPLITTESTSIZE
                        Indicates the proportion of the dataset to include in the test sets (default: 0.2)
  -splitRandomState SPLITRANDOMSTATE
                        Controls the shuffling applied to the data before applying the split (default: 0)
  -featureScaleDependentVariables
                        Indicates if the unique dependent variable should be feature scaled, if needed for the classifier (default: False)
  -predict PREDICT [PREDICT ...]
                        Defines independent variables for prediction (default: None)
  -predictOnly PREDICTONLY [PREDICTONLY ...]
                        Defines a list of classifiers for prediction (default: None, means use all existing classifiers). Here is the complete list of classifiers codes ['DTC', 'KNNC', 'KSVMC', 'LRC', 'NBC', 'RFC', 'SVMC']
  -showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]
                        Define a list of classifiers to display a comparison table of test and predictions values sets (default: None, means don't show any comparison table). Here is the complete list of classifiers codes
                        ['DTC', 'KNNC', 'KSVMC', 'LRC', 'NBC', 'RFC', 'SVMC']
  -nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW
                        Indicates the number of lines to display for the comparison table (default: 10), only applicable with -nbPredictionLinesToShow parameter
```

### 2. Display the Confusion Matrix table with the Accuracy Score descendent sorted
`python3 src/classification_model_selection.py 'src/data/Data-For-Classification.csv'`\
\
Output :
```
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|     Classification Model      |     Accuracy Score     | Number of True Positives | Number of False Positives | Number of True Negatives | Number of False Negatives | Number of True Predictions | Number of False Predictions |
+===============================+========================+==========================+===========================+==========================+===========================+============================+=============================+
| Decision Tree Classification  | 0.97080291970802923274 |            85            |             2             |            48            |             2             |            133             |              4              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      K Nearest Neighbors      | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Kernel Support Vector Machine | 0.95620437956204384911 |            82            |             5             |            49            |             1             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      Logistic Regression      | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Random Forest Classification  | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|    Support Vector Machine     | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|  Naive Bayes Classification   | 0.94890510948905104627 |            80            |             7             |            50            |             0             |            130             |              7              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
```

### 3. Check some predictions
`python3 src/classification_model_selection.py 'src/data/Data-For-Classification.csv' -showPredictionsFor DTC KNNC KSVMC LRC NBC RFC SVMC -nbPredictionLinesToShow 3`\
\
Output :
```
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|     Classification Model      |     Accuracy Score     | Number of True Positives | Number of False Positives | Number of True Negatives | Number of False Negatives | Number of True Predictions | Number of False Predictions |
+===============================+========================+==========================+===========================+==========================+===========================+============================+=============================+
| Decision Tree Classification  | 0.97080291970802923274 |            85            |             2             |            48            |             2             |            133             |              4              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      K Nearest Neighbors      | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Kernel Support Vector Machine | 0.95620437956204384911 |            82            |             5             |            49            |             1             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      Logistic Regression      | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Random Forest Classification  | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|    Support Vector Machine     | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|  Naive Bayes Classification   | 0.94890510948905104627 |            80            |             7             |            50            |             0             |            130             |              7              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+

Decision Tree Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+

K Nearest Neighbors Classification predictions comparison
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------+-----------------+
|    Sample code    |  Clump Thickness  |   Uniformity of   |   Uniformity of   | Marginal Adhesion | Single Epithelial |    Bare Nuclei    |  Bland Chromatin  |  Normal Nucleoli  |      Mitoses      | Class | Predicted Class |
|      number       |                   |     Cell Size     |    Cell Shape     |                   |     Cell Size     |                   |                   |                   |                   |       |                 |
+===================+===================+===================+===================+===================+===================+===================+===================+===================+===================+=======+=================+
| 0.124949749927309 | -1.22468404241557 | -0.69781134377775 | -0.74152574267590 | -0.63363746743558 | -0.54871997844894 | 0.427573659924659 | -0.99628733159788 | -0.62157783392617 | -0.33863738114486 |  2.0  |       2.0       |
|        44         |        25         |        23         |        28         |        52         |        48         |         7         |        06         |        76         |        774        |       |                 |
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------+-----------------+
| 0.099256755003917 | -0.51066643733398 | -0.69781134377775 | -0.74152574267590 | -0.63363746743558 | -0.54871997844894 | -0.68279598725186 | -0.59244703237557 | -0.62157783392617 | -0.33863738114486 |  2.0  |       2.0       |
|        47         |        95         |        23         |        28         |        52         |        48         |        45         |        72         |        76         |        774        |       |                 |
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------+-----------------+
| -0.56729484803143 | 0.203351167747593 | 0.603657464771018 | 0.604685580685835 | -0.27916404815297 | 0.798811242989393 | 1.815535718895314 | 0.215233566069029 | 0.023021401256525 | -0.33863738114486 |  4.0  |       4.0       |
|        85         |        63         |         9         |         9         |        297        |         2         |         7         |        82         |        16         |        774        |       |                 |
+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------+-----------------+

Kernel Support Vector Machine Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+

Logistic Regression Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+

Naive Bayes Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+

Random Forest Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+

Support Vector Machine Classification predictions comparison
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
| Sample code number |  Clump Thickness   |   Uniformity of Cell   |   Uniformity of Cell   | Marginal Adhesion  |   Single Epithelial   | Bare Nuclei | Bland Chromatin |  Normal Nucleoli   | Mitoses | Class | Predicted Class |
|                    |                    |          Size          |         Shape          |                    |       Cell Size       |             |                 |                    |         |       |                 |
+====================+====================+========================+========================+====================+=======================+=============+=================+====================+=========+=======+=================+
|     1173347.0      | 1.0000000000000004 |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     5.0     |       1.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|     1156017.0      |        3.0         |          1.0           |          1.0           | 0.9999999999999998 |          2.0          |     1.0     |       2.0       | 0.9999999999999998 |   1.0   |  2.0  |       2.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
|      706426.0      |        5.0         |          5.0           |          5.0           |        2.0         |          5.0          |    10.0     |       4.0       |        3.0         |   1.0   |  4.0  |       4.0       |
+--------------------+--------------------+------------------------+------------------------+--------------------+-----------------------+-------------+-----------------+--------------------+---------+-------+-----------------+
```

### 4. Make some predictions
`python3 src/classification_model_selection.py 'src/data/Data-For-Classification.csv' -predict 1321264 5 2 2 2 1 1 2 1 1 -predict 1331412 5 7 10 10 5 10 10 10 1`\
\
Output :
```
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|     Classification Model      |     Accuracy Score     | Number of True Positives | Number of False Positives | Number of True Negatives | Number of False Negatives | Number of True Predictions | Number of False Predictions |
+===============================+========================+==========================+===========================+==========================+===========================+============================+=============================+
| Decision Tree Classification  | 0.97080291970802923274 |            85            |             2             |            48            |             2             |            133             |              4              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      K Nearest Neighbors      | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Kernel Support Vector Machine | 0.95620437956204384911 |            82            |             5             |            49            |             1             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|      Logistic Regression      | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
| Random Forest Classification  | 0.95620437956204384911 |            84            |             3             |            47            |             3             |            131             |              6              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|    Support Vector Machine     | 0.95620437956204384911 |            83            |             4             |            48            |             2             |            131             |              6              |
|        Classification         |                        |                          |                           |                          |                           |                            |                             |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+
|  Naive Bayes Classification   | 0.94890510948905104627 |            80            |             7             |            50            |             0             |            130             |              7              |
+-------------------------------+------------------------+--------------------------+---------------------------+--------------------------+---------------------------+----------------------------+-----------------------------+

Decision Tree Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

K Nearest Neighbors Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

Kernel Support Vector Machine Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

Logistic Regression Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

Naive Bayes Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

Random Forest Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+

Support Vector Machine Classification predictions
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
| Sample code number | Clump Thickness | Uniformity of Cell Size | Uniformity of Cell Shape | Marginal Adhesion | Single Epithelial Cell Size | Bare Nuclei | Bland Chromatin | Normal Nucleoli | Mitoses | Predicted Class |
+====================+=================+=========================+==========================+===================+=============================+=============+=================+=================+=========+=================+
|     1321264.0      |       5.0       |           2.0           |           2.0            |        2.0        |             1.0             |     1.0     |       2.0       |       1.0       |   1.0   |       2.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
|     1331412.0      |       5.0       |           7.0           |           10.0           |       10.0        |             5.0             |    10.0     |      10.0       |      10.0       |   1.0   |       4.0       |
+--------------------+-----------------+-------------------------+--------------------------+-------------------+-----------------------------+-------------+-----------------+-----------------+---------+-----------------+
```

# License

## MIT License

Copyright (c) 2020 Christian Kevin TRAORÉ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
