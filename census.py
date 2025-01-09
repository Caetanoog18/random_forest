import os
import pickle
from sklearn import ensemble
from pre_processing import Census
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import accuracy_score, classification_report

class CensusDatabase:
    def __init__(self):
        self.census = Census().pre_processing()
        self.X_census_training = None
        self.y_census_training = None
        self.X_census_test = None
        self.y_census_test = None

    def census_database(self):
        file_path = 'database/census.pkl'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.X_census_training, self.y_census_training, self.X_census_test, self.y_census_test = pickle.load(file)
                print('Census Database was loaded successfully \n')
        else:
            raise FileNotFoundError('Census Database was not found. \n')

        # print(self.X_census_training.shape, self.y_census_training.shape)
        # print(self.X_census_test.shape, self.y_census_test.shape)

        random_forest = ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        random_forest.fit(self.X_census_training, self.y_census_training)

        predictions = random_forest.predict(self.X_census_test)

        # print(predictions)
        # print(self.y_census_test)

        accuracy = accuracy_score(self.y_census_test, predictions)
        print(f'Random Forest Accuracy: {accuracy:.4f} \n')

        confusion_matrix = ConfusionMatrix(random_forest)
        confusion_matrix.fit(self.X_census_training, self.y_census_training)
        confusion_matrix.score(self.X_census_test, self.y_census_test)
        confusion_matrix.show()

        print(classification_report(self.y_census_test, predictions))


