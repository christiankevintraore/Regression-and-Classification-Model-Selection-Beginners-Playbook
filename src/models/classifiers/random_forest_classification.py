"""random_forest_classification.py
~~~~~~~~~~~~~~

An implementation of Random Forest Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.ensemble import RandomForestClassifier as sklRFClassifier
from models.classifiers.generic_classifier import GenericClassifier



#### Main RandomForestClassifier class
class RandomForestClassifier(GenericClassifier):

    def evaluate(self):
        """Applies the Random Forest Classification model on the dataset.

        """
        self.classifier = sklRFClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        return self.evaluate_from_classifier('Random Forest Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Random Forest Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Random Forest Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Random Forest Classification model.

        """
        return ["Random Forest Classification predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
