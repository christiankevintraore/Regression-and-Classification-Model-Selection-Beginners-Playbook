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
        regressor = sklRandomForestRegressor(n_estimators = 10, random_state = 0)
        regressor.fit(self.datasetManager.X_train, self.datasetManager.y_train)
        self.regressor = regressor

        # Predicting the Test set results
        self.y_pred = regressor.predict(self.datasetManager.X_test)
        
        # Returning the process result : the regression type and the predicted dependent variables set
        return ["Random Forest Regression", self.get_r2_score(self.datasetManager.y_test, self.y_pred)]

    def predict(self):
        """Makes some predictions with Random Forest Regression model.

        """
        predictLambda = lambda valuesToPredict : self.regressor.predict(valuesToPredict)
        return ["Random Forest Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Random Forest Regression model.

        """
        return ["Random Forest Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
