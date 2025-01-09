import os
import pickle
from sklearn import ensemble
from pre_processing import Credit
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report

class CreditDatabase:
    def __init__(self):
        self.credit = Credit().pre_processing()
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.y_credit_training = None
        self.X_credit_test = None
        self.y_credit_test = None

    def credit_database(self):
        file_path = 'database/credit.pkl'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = pickle.load(file)
                print('Credit database was loaded successfully \n')
        else:
            raise FileNotFoundError('Credit database was not found. \n')

        # print(self.X_credit_training.shape, self.y_credit_training.shape)
        # print(self.X_credit_test.shape, self.y_credit_test.shape)

        random_forest_credit = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
        random_forest_credit.fit(self.X_credit_training, self.y_credit_training)

        predictions = random_forest_credit.predict(self.X_credit_test)

        # print(predictions)
        # print(self.y_credit_test)

        accuracy = accuracy_score(self.y_credit_test, predictions)
        print(f'Random Forest accuracy: {accuracy:.4f} \n')

        confusion_matrix = ConfusionMatrix(random_forest_credit)
        confusion_matrix.fit(self.X_credit_training, self.y_credit_training)
        confusion_matrix.score(self.X_credit_test, self.y_credit_test)
        confusion_matrix.show()

        print(classification_report(self.y_credit_test, predictions))






