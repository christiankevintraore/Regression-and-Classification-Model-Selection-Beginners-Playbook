"""support_vector_regression.py
~~~~~~~~~~~~~~

An implementation of Support Vector Regression (SVR).

Desirable features :
    - Tune the regressor input parameters for better performance.
    - Auto detection of 'featureScaleDependentVariables' -> delete the concerned input parameter.

"""

#### Libraries
from sklearn.svm import SVR
import utils.dataset_manager as dm
import models.regressors.generic_regressor as gr



#### Main SupportVectorRegressor class
class SupportVectorRegressor(gr.GenericRegressor):

    def evaluate(self):
        """Applies the SVR model on the dataset.

        """
        # Method variables definition
        X_train, X_test, y_train, y_test = dm.reshape_y_set_split_data(self.datasetManager)
        featureScaleDependentVariables = self.datasetManager.params.featureScaleDependentVariables

        # Feature Scaling
        X_scaler, X_train = dm.do_feature_scaling(X_train)
        if featureScaleDependentVariables:
            y_scaler, y_train = dm.do_feature_scaling(y_train)
        else:
            y_scaler = None
            y_train = self.datasetManager.y_train
        
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler

        # Training the SVR model on the training set
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train.ravel())
        self.regressor = regressor

        # Predicting the Test set results
        self.y_pred = y_scaler.inverse_transform(regressor.predict(X_scaler.transform(X_test))) if featureScaleDependentVariables else regressor.predict(X_test)
        
        # Returning the process result : the regression type and the predicted dependent variables set
        return ["Support Vector Regression", self.get_r2_score(y_test, self.y_pred)]

    def predict(self):
        """Makes some predictions with Support Vector Regression model.

        """
        predictLambda = lambda valuesToPredict : self.y_scaler.inverse_transform(self.regressor.predict(self.X_scaler.transform(valuesToPredict)))\
            if self.datasetManager.params.featureScaleDependentVariables else self.regressor.predict(valuesToPredict)
        
        return ["Support Vector Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Support Vector Regression model.

        """
        return ["Support Vector Regression predictions comparison", super().truncate_predictions_relevance(self.datasetManager.X_test, self.datasetManager.y_test, self.y_pred)]
