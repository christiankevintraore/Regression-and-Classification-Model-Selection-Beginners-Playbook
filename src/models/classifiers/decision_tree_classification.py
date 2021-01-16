"""decision_tree_classification.py
~~~~~~~~~~~~~~

An implementation of Decision Tree Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.tree import DecisionTreeClassifier as skDTC
from models.classifiers.generic_classifier import GenericClassifier



#### Main DecisionTreeClassifier class
class DecisionTreeClassifier(GenericClassifier):

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
        return ["Decision Tree Classification predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
