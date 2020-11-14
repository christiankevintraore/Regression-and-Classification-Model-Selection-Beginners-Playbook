"""generic_classifier.py
~~~~~~~~~~~~~~

A generic classifier that will be extends by each classifier.
It's also implements commons util methods used by classifiers.

"""

#### Libraries
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import models.common.common_model_selection as cms
import utils.dataset_manager as dm



#### Main GenericClassifier class
class GenericClassifier(cms.CommonModelSelection):

    def __init__(self, datasetManager:dm.DatasetManager):
        """Initialize the current classifier object with the dataset manager.
        It also do feature scaling on X_train and X_test.

        """
        super().__init__(datasetManager)
        
        X_scaler, X_train = dm.do_feature_scaling(datasetManager.X_train)
        X_test = X_scaler.transform(datasetManager.X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.X_scaler = X_scaler

    def evaluate_from_classifier(self, classificationName, classifier):
        """A common method for training and predicting X_test with the provided classifier.

        """
        # Training the classifier on the Training set
        classifier.fit(self.X_train, self.datasetManager.y_train)
        
        # Predicting the Test set results
        self.y_pred = classifier.predict(self.X_test)
        
        # Returning the process result : the classifier type, the confusion matrix and the accuracy score
        return [classificationName] + self.get_confusion_matrix_and_accuracy_score(self.datasetManager.y_test, self.y_pred)

    def get_confusion_matrix_and_accuracy_score(self, y_test:pd.DataFrame, y_pred:pd.DataFrame):
        """Evaluates a classifier model performance with the y_test and y_pred DataFrame inputs, and returns an array of Confusion Matrix and Accuracy Score.

        """
        return [confusion_matrix(y_test, y_pred), accuracy_score(y_test, y_pred)]
