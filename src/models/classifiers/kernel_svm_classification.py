"""kernel_svm_classification.py
~~~~~~~~~~~~~~

An implementation of Kernel Support Vector Machine Classification (Kernel SVM).

Desirable features :
    - Tune the classifier input parameters for better performance.

"""

#### Libraries
from sklearn.svm import SVC
from models.classifiers.generic_classifier import GenericClassifier



#### Main KernelSvmClassifier class
class KernelSvmClassifier(GenericClassifier):

    def evaluate(self):
        """Applies the Kernel Support Vector Machine Classification model on the dataset.

        """
        self.classifier = SVC(kernel = 'rbf', random_state = 0)
        return self.evaluate_from_classifier('Kernel Support Vector Machine Classification', self.classifier)

    def predict(self):
        """Makes some predictions with Kernel Support Vector Machine Classification model.

        """
        predictLambda = lambda valuesToPredict : self.classifier.predict(self.X_scaler.transform(valuesToPredict))
        return ["Kernel Support Vector Machine Classification predictions", super().predict_user_input_variables(predictLambda)]



    def predictions_relevance(self):
        """Returns a comparison table for Kernel Support Vector Machine Classification model.

        """
        return ["Kernel Support Vector Machine Classification predictions comparison",\
            super().get_predictions_relevance(self.datasetManager.X_test_for_predictions_relevance,\
                self.datasetManager.y_test_for_predictions_relevance, self.y_pred)]
