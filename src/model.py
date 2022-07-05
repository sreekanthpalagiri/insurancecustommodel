from sklearn.base import TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import ClassifierMixin
from joblib import load
from sklearn.pipeline import Pipeline


class TransformData(TransformerMixin):

    def bmi(self, wt, ht):
        ht_m = (((ht//100)*12+ht % 100)*0.0254)
        wt_kg = wt/2.20462
        bmi = wt_kg/(ht_m**2)
        return bmi

    def age_group(self, age):
        if age < 18:
            return 0
        elif age < 40:
            return 1
        elif age <= 60:
            return 2
        else:
            return 3

    def quote_value(gender, bmi, age_group):
        if (bmi > 17.49 and bmi < 38.5):
            base_value = 500
        else:
            if age_group == 0:
                base_value = 500
            elif age_group == 1:
                base_value = 750
            elif age_group == 2:
                base_value = 1000
            else:
                base_value = 2000
        if gender == 0:
            return base_value * 0.9
        else:
            return base_value

    def encode(self):
        enc = OrdinalEncoder()
        enc.fit(self.X[['Ins_Gender']])
        self.X[['Ins_Gender']] = enc.transform(self.X[['Ins_Gender']])

    def transform(self, X):
        self.X = X
        self.X['bmi'] = self.X.apply(lambda x: self.bmi(x.Wt, x.Ht), axis=1)
        self.X['age_group'] = self.X.apply(
            lambda x: self.age_group(x.Ins_Age), axis=1)

        self.X.drop(columns=['Ht', 'Wt', 'IssueDate', 'Ins_Age'], inplace=True)

        self.encode()

        return self.X


class InsuranceModel(ClassifierMixin):
    def __init__(self):
        self.clf = load('./model/randomforest.joblib')

    def fit(self, X):
        pass

    def predict(self, X):
        self.X = X
        self.X['quote'] = self.clf.predict(
            X[['Ins_Gender', 'bmi', 'age_group']].to_numpy())
        return self.X[['AppID', 'quote']]


pipe = Pipeline([('transformer', TransformData()),
                 ('InsuranceModel', InsuranceModel())])
