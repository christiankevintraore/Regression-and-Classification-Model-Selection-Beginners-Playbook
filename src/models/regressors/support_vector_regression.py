"""support_vector_regression.py
~~~~~~~~~~~~~~

An implementation of Support Vector Regression (SVR).

Desirable features :
    - Tune the regressor input parameters for better performance.
    - Auto detection of 'featureScaleDependentVariables' -> delete the concerned input parameter.

"""

#### Libraries
from sklearn.svm import SVR
from dataset.model_selection_dataset_manager import reshape_y_set_split_data, do_feature_scaling
from models.regressors.generic_regressor import GenericRegressor



#### Main SupportVectorRegressor class
class SupportVectorRegressor(GenericRegressor):

    def evaluate(self):
        """Applies the SVR model on the dataset.

        """
        # Method variables definition
        X_train, X_test, y_train, y_test = reshape_y_set_split_data(self.datasetManager)
        featureScaleDependentVariables = self.datasetManager.params.featureScaleDependentVariables

        # Feature Scaling
        self.X_scaler, X_train = do_feature_scaling(X_train)
        if featureScaleDependentVariables:
            self.y_scaler, y_train = do_feature_scaling(y_train)
        else:
            self.y_scaler = None
            y_train = self.datasetManager.y_train
        
        # Training the SVR model on the training set
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train.to_numpy().ravel())
        self.regressor = regressor

        # Predicting the Test set results
        self.y_pred = self.y_scaler.inverse_transform(regressor.predict(self.X_scaler.transform(X_test)))\
            if featureScaleDependentVariables else regressor.predict(X_test)
        
        # Returning the process result : the regression type and the predicted dependent variables set
        return ["Support Vector Regression", self.get_r2_score(y_test, self.y_pred)]

    def predict(self):
        """Makes some predictions with Support Vector Regression model.

        """
        from utils.common_utils import flatten
        predictLambda = lambda valuesToPredict : flatten(self.y_scaler.inverse_transform(self.regressor.\
            predict(self.X_scaler.transform(valuesToPredict)))) if self.datasetManager.params.featureScaleDependentVariables\
                else flatten(self.regressor.predict(valuesToPredict))
        
        return ["Support Vector Regression predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Support Vector Regression model.

        """
        return ["Support Vector Regression predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
