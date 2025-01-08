import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class CreditRisk:
    def __init__(self):
        self.credit_risk = pd.read_csv('database/credit_risk.csv')
        self.X_credit_risk = None
        self.y_credit_risk = None

    def pre_processing(self):

        self.X_credit_risk = self.credit_risk.iloc[:, 0:4].values
        self.y_credit_risk = self.credit_risk.iloc[:, 4].values

        label_encoder_history = LabelEncoder()
        label_encoder_debt = LabelEncoder()
        label_encoder_guarantee = LabelEncoder()
        label_encoder_income = LabelEncoder()

        self.X_credit_risk[:, 0] = label_encoder_history.fit_transform(self.X_credit_risk[:, 0])
        self.X_credit_risk[:, 1] = label_encoder_debt.fit_transform(self.X_credit_risk[:, 1])
        self.X_credit_risk[:, 2] = label_encoder_guarantee.fit_transform(self.X_credit_risk[:, 2])
        self.X_credit_risk[:, 3] = label_encoder_income.fit_transform(self.X_credit_risk[:, 3])

        with open(r'database/credit_risk.pkl', 'wb') as file:
            pickle.dump([self.X_credit_risk, self.y_credit_risk], file)


class Credit:
    def __init__(self):
        self.credit = pd.read_csv('database/credit_data.csv')
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.X_credit_test = None
        self.y_credit_training = None
        self.y_credit_test = None

        # Verifying inconsistent values
        # print(self.credit.loc[self.credit['age']<0])

    def pre_processing(self):
        # Mean without inconsistent values
        mean = self.credit['age'][self.credit['age']>0].mean()

        self.credit.loc[self.credit['age']<0, 'age'] = mean

        #Missing values
        # print(self.credit.loc[pd.isnull(self.credit['age'])])
        self.credit.fillna(self.credit['age'].mean(), inplace=True)

        self.X_credit = self.credit.iloc[:, 1:4].values
        self.y_credit = self.credit.iloc[:, 4].values

        # Printing the maximum and the minimum values
        # print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        # print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        credit_scaler = StandardScaler()
        self.X_credit = credit_scaler.fit_transform(self.X_credit)

        # Printing the maximum and the minimum values
        # print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        # print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = train_test_split(self.X_credit, self.y_credit, test_size=0.25, random_state=42)

        with open('database/credit.pkl', 'wb') as file:
            pickle.dump([self.X_credit_training,self.X_credit_test,self.y_credit_training, self.y_credit_test], file)

class Census:
    def __init__(self):
        self.census = pd.read_csv('database/census.csv')
        self.X_census = None
        self.y_census = None
        self.X_census_training = None
        self.X_census_test = None
        self.y_census_training = None
        self.y_census_test = None

    def pre_processing(self):
        self.X_census = self.census.iloc[:, 0:14].values
        self.y_census = self.census.iloc[:, 14].values


        # label_encoder_workclass = LabelEncoder()
        # label_encoder_education = LabelEncoder()
        # label_encoder_marital = LabelEncoder()
        # label_encoder_occupation = LabelEncoder()
        # label_encoder_relationship = LabelEncoder()
        # label_encoder_race = LabelEncoder()
        # label_encoder_sex = LabelEncoder()
        # label_encoder_country = LabelEncoder()

        # self.X_census[:, 1] = label_encoder_workclass.fit_transform(self.X_census[:,1])
        # self.X_census[:, 3] = label_encoder_education.fit_transform(self.X_census[:,3])
        # self.X_census[:, 5] = label_encoder_marital.fit_transform(self.X_census[:,5])
        # self.X_census[:, 6] = label_encoder_occupation.fit_transform(self.X_census[:,6])
        # self.X_census[:, 7] = label_encoder_relationship.fit_transform(self.X_census[:,7])
        # self.X_census[:, 8] = label_encoder_race.fit_transform(self.X_census[:,8])
        # self.X_census[:, 9] = label_encoder_sex.fit_transform(self.X_census[:,9])
        # self.X_census[:, 13] = label_encoder_country.fit_transform(self.X_census[:,13])

        one_hot_encoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
        self.X_census = one_hot_encoder.fit_transform(self.X_census).toarray()

        scaler = StandardScaler()
        self.X_census = scaler.fit_transform(self.X_census)

        self.X_census_training, self.X_census_test, self.y_census_training, self.y_census_test = train_test_split(self.X_census, self.y_census, test_size=0.15, random_state=42)

        # print(self.X_census_training.shape, self.y_census_training.shape)
        # print(self.X_census_test.shape, self.y_census_test.shape)

        with open('database/census.pkl', 'wb') as file:
            pickle.dump([self.X_census_training, self.y_census_training, self.X_census_test, self.y_census_test], file)