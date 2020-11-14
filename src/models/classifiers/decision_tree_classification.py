"""decision_tree_classification.py
~~~~~~~~~~~~~~

An implementation of Decision Tree Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.tree import DecisionTreeClassifier as skDTC
import models.classifiers.generic_classifier as gc



#### Main DecisionTreeClassifier class
class DecisionTreeClassifier(gc.GenericClassifier):

    def evaluate(self):
        """Applies the Decision Tree Classification model on the dataset.

        """
        self.classifier = skDTC(criterion = 'entropy', random_state = 0)
        return self.evaluate_from_classifier('Decision Tree Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Decision Tree Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Decision Tree Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Decision Tree Classification model.

        """
        return ["Decision Tree Classification predictions comparison", super().truncate_predictions_relevance(self.X_scaler.inverse_transform(self.X_test), self.datasetManager.y_test, self.y_pred)]
