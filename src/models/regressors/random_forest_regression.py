"""random_forest_regression.py
~~~~~~~~~~~~~~

An implementation of Random Forest Regression (RFR).

Desirable features :
    - Tune the regressor input parameters for better performance.

"""

#### Libraries
from sklearn.ensemble import RandomForestRegressor as sklRandomForestRegressor
import models.regressors.generic_regressor as gr



#### Main RandomForestRegressor class
class RandomForestRegressor(gr.GenericRegressor):

    def evaluate(self):
        """Applies the Random Forest Regression model on the dataset.

        """
        # Training the Random Forest Regression model on the Training set
        self.regressor = sklRandomForestRegressor(n_estimators = 10, random_state = 0)
        return self.evaluate_from_dataset_manager_and_regressor("Random Forest Regression", self.regressor)

    def predict(self):
        """Makes some predictions with Random Forest Regression model.

        """
        predictLambda = lambda valuesToPredict : self.regressor.predict(valuesToPredict)
        return ["Random Forest Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Random Forest Regression model.

        """
        return ["Random Forest Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
