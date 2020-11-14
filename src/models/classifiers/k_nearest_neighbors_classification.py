"""k_nearest_neighbors_classification.py
~~~~~~~~~~~~~~

An implementation of K Nearest Neighbors Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.neighbors import KNeighborsClassifier
import models.classifiers.generic_classifier as gc



#### Main KNearestNeighborsClassifier class
class KNearestNeighborsClassifier(gc.GenericClassifier):

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
        return ["K Nearest Neighbors Classification predictions comparison", super().truncate_predictions_relevance(self.X_test, self.datasetManager.y_test, self.y_pred)]
