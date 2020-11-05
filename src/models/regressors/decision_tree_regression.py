"""decision_tree_regression.py
~~~~~~~~~~~~~~

An implementation of Decision Tree Regression.

Desirable features :
    - Tune the regressor input parameters for better performance.

"""

#### Libraries
from sklearn.tree import DecisionTreeRegressor as sklDecisionTreeRegressor
import models.regressors.generic_regressor as gr



#### Main DecisionTreeRegressor class
class DecisionTreeRegressor(gr.GenericRegressor):

    def evaluate(self):
        """Applies the Decision Tree Regression model on the dataset.

        """
        # Training the Decision Tree Regression model on the Training set
        regressor = sklDecisionTreeRegressor(random_state = 0)
        regressor.fit(self.datasetManager.X_train, self.datasetManager.y_train)
        self.regressor = regressor

        # Predicting the Test set results
        self.y_pred = regressor.predict(self.datasetManager.X_test)
        
        # Returning the process result : the regression type and the predicted dependent variables set
        return ["Decision Tree Regression", self.get_r2_score(self.datasetManager.y_test, self.y_pred)]

    def predict(self):
        """Makes some predictions with Decision Tree Regression model.

        """
        predictLambda = lambda valuesToPredict : self.regressor.predict(valuesToPredict)
        return ["Decision Tree Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Decision Tree Regression model.

        """
        return ["Decision Tree Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
