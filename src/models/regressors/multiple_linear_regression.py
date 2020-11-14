"""multiple_linear_regression.py
~~~~~~~~~~~~~~

An implementation of Multiple Linear Regression (MLR).

Desirable features :
    - Tune the regressor input parameters for better performance.

"""

#### Libraries
from sklearn.linear_model import LinearRegression
import models.regressors.generic_regressor as gr



#### Main MultipleLinearRegressor class
class MultipleLinearRegressor(gr.GenericRegressor):

    def evaluate(self):
        """Applies the Multiple Linear Regression model on the dataset.

        """
        # Training the Multiple Linear Regression model on the Training set
        self.regressor = LinearRegression()
        return self.evaluate_from_dataset_manager_and_regressor("Multiple Linear Regression", self.regressor)
        
    def predict(self):
        """Makes some predictions with Multiple Linear Regression model.

        """
        predictLambda = lambda valuesToPredict : self.regressor.predict(valuesToPredict)
        return ["Multiple Linear Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Multiple Linear Regression model.

        """
        return ["Multiple Linear Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
