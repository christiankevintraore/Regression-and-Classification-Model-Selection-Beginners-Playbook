# Regression and Classification model selection playbook

## A beginners playbook to determine the best regression or classification model applied to a dataset.

### The goal of this project is to automate Regression and Classification model selection through a simple command line.

This tool was inspired by the great course of [Machine Learning A-Z™: Hands-On Python & R In Data Science](https://udemy.com/course/machinelearning/), made by [Kirill EREMENKO](https://udemy.com/user/kirilleremenko/) and [Hadelin DE PONTEVES](https://udemy.com/user/hadelin-de-ponteves/).\
Thanks a lot, I really enjoyed it -:)\
Also notice that the datasets, for test purpose, were took from the provided resources of this course.

## Dataset Template
* *To work properly with this tool, your dataset should have consecutive independent variable columns ; and only one dependant variable column.*
* *It is also a good idea to always have headers, it helps to identify columns on printed tables.*

## Required python modules
Depending of modules already installed in your environment ; you may have to install one or several of these modules :
* Scikit-Learn :  ```pip install -U scikit-learn```
* TextTable :     ```pip install -U texttable```
* Numpy :         ```pip install -U numpy```
* Pandas :        ```pip install -U pandas```

## Command samples
From the root directory of this project :
# 1. Display the helper
`python3 src/regression_model_selection.py -h`\
\
Output :
```
usage: regression_model_selection.py [-h] [-noHeader] [-dependentVariableColumnIndex DEPENDENTVARIABLECOLUMNINDEX]
                                     [-independentVariablesStartIndex INDEPENDENTVARIABLESSTARTINDEX] [-independentVariablesEndIndex INDEPENDENTVARIABLESENDINDEX]
                                     [-splitTestSize SPLITTESTSIZE] [-splitRandomState SPLITRANDOMSTATE] [-featureScaleDependentVariables]
                                     [-predict PREDICT [PREDICT ...]] [-predictOnly PREDICTONLY [PREDICTONLY ...]]
                                     [-showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]] [-nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW]
                                     dataset

Determine the best regressor type to use for a specific dataset.

positional arguments:
  dataset               A dataset file path

optional arguments:
  -h, --help            show this help message and exit
  -noHeader             Indicates that there is no header in the dataset (default: False)
  -dependentVariableColumnIndex DEPENDENTVARIABLECOLUMNINDEX
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
                        Defines a list of regressors for prediction (default: None, means use all existing regressors). Here is the complete list of regressors codes
                        ['MLR', 'POLY', 'SVR', 'DTR', 'RFR']
  -showPredictionsFor SHOWPREDICTIONSFOR [SHOWPREDICTIONSFOR ...]
                        Define a list of regressors to display a comparison table of test and predictions values sets (default: None, means don't show any comparison
                        table). Here is the complete list of regressors codes ['MLR', 'POLY', 'SVR', 'DTR', 'RFR']
  -nbPredictionLinesToShow NBPREDICTIONLINESTOSHOW
                        Indicates the number of lines to display for the comparison table (default: 10), only applicable with -nbPredictionLinesToShow parameter
```

# 2. Display the R2 score table descendent sorted
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv'`\
\
Remember to always add the *-featureScaleDependentVariables* option to indicate that the dependent variables column should be feature scaled. There is not yet an auto detection for that\
It is the minimum output of this script.\
\
Output :
```
+----------------------------+------------------------+
|      Regression Type       |        R2 Score        |
+============================+========================+
| Random Forest Regression   | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
| Polynomial Regression      | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
| Decision Tree Regression   | 0.92290587417794101022 |
+----------------------------+------------------------+
```

# 3. See some predictions
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv' -featureScaleDependentVariables -showPredictionsFor MLR POLY SVR DTR RFR -nbPredictionLinesToShow 3`\
\
Output :
```
+----------------------------+------------------------+
|      Regression Type       |        R2 Score        |
+============================+========================+
| Random Forest Regression   | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
| Polynomial Regression      | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
| Decision Tree Regression   | 0.92290587417794101022 |
+----------------------------+------------------------+

      Multiple Linear Regression predictions comparison
+--------+--------+----------+--------+---------+--------------+
|   AT   |   V    |    AP    |   RH   |   PE    | Predicted PE |
+========+========+==========+========+=========+==============+
| 28.660 | 77.950 | 1009.560 | 69.070 | 431.230 | 431.428      |
+--------+--------+----------+--------+---------+--------------+
| 17.480 | 49.390 | 1021.510 | 84.530 | 460.010 | 458.561      |
+--------+--------+----------+--------+---------+--------------+
| 14.860 | 43.140 | 1019.210 | 99.140 | 461.140 | 462.753      |
+--------+--------+----------+--------+---------+--------------+

      Polynomial Regression predictions comparison
+--------+--------+----------+--------+---------+--------------+
|   AT   |   V    |    AP    |   RH   |   PE    | Predicted PE |
+========+========+==========+========+=========+==============+
| 28.660 | 77.950 | 1009.560 | 69.070 | 431.230 | 433.944      |
+--------+--------+----------+--------+---------+--------------+
| 17.480 | 49.390 | 1021.510 | 84.530 | 460.010 | 457.905      |
+--------+--------+----------+--------+---------+--------------+
| 14.860 | 43.140 | 1019.210 | 99.140 | 461.140 | 460.525      |
+--------+--------+----------+--------+---------+--------------+

      Support Vector Regression predictions comparison
+--------+--------+----------+--------+---------+--------------+
|   AT   |   V    |    AP    |   RH   |   PE    | Predicted PE |
+========+========+==========+========+=========+==============+
| 28.660 | 77.950 | 1009.560 | 69.070 | 431.230 | 434.052      |
+--------+--------+----------+--------+---------+--------------+
| 17.480 | 49.390 | 1021.510 | 84.530 | 460.010 | 457.938      |
+--------+--------+----------+--------+---------+--------------+
| 14.860 | 43.140 | 1019.210 | 99.140 | 461.140 | 461.031      |
+--------+--------+----------+--------+---------+--------------+

      Decision Tree Regression predictions comparison
+--------+--------+----------+--------+---------+--------------+
|   AT   |   V    |    AP    |   RH   |   PE    | Predicted PE |
+========+========+==========+========+=========+==============+
| 28.660 | 77.950 | 1009.560 | 69.070 | 431.230 | 431.280      |
+--------+--------+----------+--------+---------+--------------+
| 17.480 | 49.390 | 1021.510 | 84.530 | 460.010 | 459.590      |
+--------+--------+----------+--------+---------+--------------+
| 14.860 | 43.140 | 1019.210 | 99.140 | 461.140 | 460.060      |
+--------+--------+----------+--------+---------+--------------+

      Random Forest Regression predictions comparison
+--------+--------+----------+--------+---------+--------------+
|   AT   |   V    |    AP    |   RH   |   PE    | Predicted PE |
+========+========+==========+========+=========+==============+
| 28.660 | 77.950 | 1009.560 | 69.070 | 431.230 | 434.049      |
+--------+--------+----------+--------+---------+--------------+
| 17.480 | 49.390 | 1021.510 | 84.530 | 460.010 | 458.785      |
+--------+--------+----------+--------+---------+--------------+
| 14.860 | 43.140 | 1019.210 | 99.140 | 461.140 | 463.020      |
+--------+--------+----------+--------+---------+--------------+
```

# 4. Make some predictions
`python3 src/regression_model_selection.py 'src/data/Data-For-Regression.csv' -featureScaleDependentVariables -predict 23.04 59.43 1010.23 68.99 -predict 18.5 51.43 1010.82 92.04`\
\
Output :
```
+----------------------------+------------------------+
|      Regression Type       |        R2 Score        |
+============================+========================+
| Random Forest Regression   | 0.96159083343638762642 |
+----------------------------+------------------------+
| Support Vector Regression  | 0.94807840499862583439 |
+----------------------------+------------------------+
| Polynomial Regression      | 0.94581932263413770468 |
+----------------------------+------------------------+
| Multiple Linear Regression | 0.93253155547613031384 |
+----------------------------+------------------------+
| Decision Tree Regression   | 0.92290587417794101022 |
+----------------------------+------------------------+

      Multiple Linear Regression predictions
+--------+--------+----------+--------+--------------+
|   AT   |   V    |    AP    |   RH   | Predicted PE |
+========+========+==========+========+==============+
| 23.040 | 59.430 | 1010.230 | 68.990 | 446.952      |
+--------+--------+----------+--------+--------------+
| 18.500 | 51.430 | 1010.820 | 92.040 | 454.196      |
+--------+--------+----------+--------+--------------+

      Polynomial Regression predictions
+--------+--------+----------+--------+--------------+
|   AT   |   V    |    AP    |   RH   | Predicted PE |
+========+========+==========+========+==============+
| 23.040 | 59.430 | 1010.230 | 68.990 | 444.775      |
+--------+--------+----------+--------+--------------+
| 18.500 | 51.430 | 1010.820 | 92.040 | 453.532      |
+--------+--------+----------+--------+--------------+

      Support Vector Regression predictions
+--------+--------+----------+--------+--------------+
|   AT   |   V    |    AP    |   RH   | Predicted PE |
+========+========+==========+========+==============+
| 23.040 | 59.430 | 1010.230 | 68.990 | 445.197      |
+--------+--------+----------+--------+--------------+
| 18.500 | 51.430 | 1010.820 | 92.040 | 452.830      |
+--------+--------+----------+--------+--------------+

      Decision Tree Regression predictions
+--------+--------+----------+--------+--------------+
|   AT   |   V    |    AP    |   RH   | Predicted PE |
+========+========+==========+========+==============+
| 23.040 | 59.430 | 1010.230 | 68.990 | 442.990      |
+--------+--------+----------+--------+--------------+
| 18.500 | 51.430 | 1010.820 | 92.040 | 459.420      |
+--------+--------+----------+--------+--------------+

      Random Forest Regression predictions
+--------+--------+----------+--------+--------------+
|   AT   |   V    |    AP    |   RH   | Predicted PE |
+========+========+==========+========+==============+
| 23.040 | 59.430 | 1010.230 | 68.990 | 442.990      |
+--------+--------+----------+--------+--------------+
| 18.500 | 51.430 | 1010.820 | 92.040 | 456.378      |
+--------+--------+----------+--------+--------------+
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
