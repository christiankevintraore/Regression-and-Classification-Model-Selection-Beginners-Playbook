"""k_nearest_neighbors_classification.py
~~~~~~~~~~~~~~

An implementation of K Nearest Neighbors Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.
    - n_neighbors == int(sqrt(self.NumberOfSamples)) ??

"""

#### Libraries
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from models.classifiers.generic_classifier import GenericClassifier



#### Main KNearestNeighborsClassifier class
class KNearestNeighborsClassifier(GenericClassifier):

    def evaluate(self):
        """Applies the K Nearest Neighbors Classification model on the dataset.

        """
        self.classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        return self.evaluate_from_classifier('K Nearest Neighbors Classification', self.classifier)

    def predict(self):
        """Makes some predictions with K Nearest Neighbors Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(valuesToPredict)
        return ["K Nearest Neighbors Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for K Nearest Neighbors Classification model.

        """
        return ["K Nearest Neighbors Classification predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
