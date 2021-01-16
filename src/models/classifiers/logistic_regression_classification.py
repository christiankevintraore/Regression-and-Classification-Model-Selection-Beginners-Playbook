"""logistic_regression_classification.py
~~~~~~~~~~~~~~

An implementation of Logistic Regression Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.linear_model import LogisticRegression
from models.classifiers.generic_classifier import GenericClassifier



#### Main LogisticRegressionClassifier class
class LogisticRegressionClassifier(GenericClassifier):

    def evaluate(self):
        """Applies the Logistic Regression Classification model on the dataset.

        """
        self.classifier = LogisticRegression(random_state = 0)
        return self.evaluate_from_classifier('Logistic Regression Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Logistic Regression Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Logistic Regression Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Logistic Regression Classification model.

        """
        return ["Logistic Regression Classification predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
