"""multiple_linear_regression.py
~~~~~~~~~~~~~~

An implementation of Multiple Linear Regression (MLR).

Desirable features :
    - Tune the regressor input parameters for better performance.

"""

#### Libraries
from sklearn.linear_model import LinearRegression
from models.regressors.generic_regressor import GenericRegressor



#### Main MultipleLinearRegressor class
class MultipleLinearRegressor(GenericRegressor):

    def evaluate(self):
        """Applies the Multiple Linear Regression model on the dataset.

        """
        # Training the Multiple Linear Regression model on the Training set
        self.regressor = LinearRegression()
        return self.evaluate_from_dataset_manager_and_regressor("Multiple Linear Regression", self.regressor)
        
    def predict(self):
        """Makes some predictions with Multiple Linear Regression model.

        """
        from utils.common_utils import flatten
        predictLambda = lambda valuesToPredict : flatten(self.regressor.predict(valuesToPredict))
        return ["Multiple Linear Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Multiple Linear Regression model.

        """
        return ["Multiple Linear Regression predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
