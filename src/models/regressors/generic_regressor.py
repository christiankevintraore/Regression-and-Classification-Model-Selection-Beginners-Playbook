"""generic_regressor.py
~~~~~~~~~~~~~~

A generic regressor that will be extends by each regressor.
It's also implements common util methods used by regressors.

"""

#### Libraries
from pandas import DataFrame
from sklearn.metrics import r2_score
from models.common.common_model_selection import CommonModelSelection



#### Main GenericRegressor class
class GenericRegressor(CommonModelSelection):

    def evaluate_from_dataset_manager_and_regressor(self, regressorName, regressor):
        """A common method for training and predicting X_test with the provided regressor.

        """
        # Training the regressor on the Training set
        regressor.fit(self.datasetManager.X_train, self.datasetManager.y_train.to_numpy().ravel())

        # Predicting the Test set results
        self.y_pred = regressor.predict(self.datasetManager.X_test)
        
        # Returning the process result : the regression name and the R2 score
        return [regressorName, self.get_r2_score(self.datasetManager.y_test, self.y_pred)]

    def get_r2_score(self, y_test:DataFrame, y_pred:DataFrame):
        """Evaluates a regressor model performance with the y_test and y_pred DataFrame inputs, and returns the R2 score.

        """
        return r2_score(y_test, y_pred)
