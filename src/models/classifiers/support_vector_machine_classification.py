"""support_vector_machine_classification.py
~~~~~~~~~~~~~~

An implementation of Support Vector Machine Classification.

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.svm import SVC
import models.classifiers.generic_classifier as gc



#### Main SupportVectorMachineClassifier class
class SupportVectorMachineClassifier(gc.GenericClassifier):

    def evaluate(self):
        """Applies the Support Vector Machine Classification model on the dataset.

        """
        self.classifier = SVC(kernel = 'linear', random_state = 0)
        return self.evaluate_from_classifier('Support Vector Machine Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Support Vector Machine Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Support Vector Machine Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Support Vector Machine Classification model.

        """
        return ["Support Vector Machine Classification predictions comparison", super().truncate_predictions_relevance(self.X_scaler.inverse_transform(self.X_test), self.datasetManager.y_test, self.y_pred)]
