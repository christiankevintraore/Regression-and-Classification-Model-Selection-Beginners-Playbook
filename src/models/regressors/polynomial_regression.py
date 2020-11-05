"""polynomial_regression.py
~~~~~~~~~~~~~~

An implementation of Polynomial Regression.

Desirable features :
    - Tune the regressor input parameters for better performance.

"""

#### Libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import models.regressors.generic_regressor as gr



#### Main PolynomialRegressor class
class PolynomialRegressor(gr.GenericRegressor):

    def evaluate(self):
        """Applies the Polynomial Regression model on the dataset.

        """
        # Training the Polynomial Regression model on the Training set
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(self.datasetManager.X_train)
        self.poly_reg = poly_reg

        regressor = LinearRegression()
        regressor.fit(X_poly, self.datasetManager.y_train)
        self.regressor = regressor

        # Predicting the Test set results
        self.y_pred = regressor.predict(poly_reg.transform(self.datasetManager.X_test))
        
        # Returning the process result : the regression type and the predicted dependent variables set
        return ["Polynomial Regression", self.get_r2_score(self.datasetManager.y_test, self.y_pred)]

    def predict(self):
        """Makes some predictions with Polynomial Regression model.

        """
        predictLambda = lambda valuesToPredict : self.regressor.predict(self.poly_reg.transform(valuesToPredict))
        return ["Polynomial Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Polynomial Regression model.

        """
        return ["Polynomial Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
