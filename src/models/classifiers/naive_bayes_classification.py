"""naive_bayes_classification.py
~~~~~~~~~~~~~~

An implementation of Naive Bayes Classification.

Desirable features :
    - Tune the classifier input parameters for better performance (change classifier type ?).

"""

#### Libraries
from sklearn.naive_bayes import GaussianNB
import models.classifiers.generic_classifier as gc



#### Main NaiveBayesClassifier class
class NaiveBayesClassifier(gc.GenericClassifier):

    def evaluate(self):
        """Applies the Naive Bayes Classification model on the dataset.

        """
        self.classifier = GaussianNB()
        return self.evaluate_from_classifier('Naive Bayes Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Naive Bayes Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Naive Bayes Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Naive Bayes Classification model.

        """
        return ["Naive Bayes Classification predictions comparison", super().truncate_predictions_relevance(self.X_scaler.inverse_transform(self.X_test), self.datasetManager.y_test, self.y_pred)]
