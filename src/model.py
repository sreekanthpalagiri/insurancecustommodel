from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline

class TransformData(TransformerMixin):
    def fit(self, X, y=None):
        self.X = X   
        
    def bmi_healthy(self,wt,ht):
        ht_m=(((ht//100)*12+ht%100)*0.0254)
        wt_kg=wt/2.20462
        bmi=wt_kg/(ht_m**2)
        return True if (bmi > 17.49 and bmi < 38.5) else False
    
    def age_group(self,age):
        if age < 18:
            return 0
        elif age < 40:
            return 1
        elif age <= 60:
            return 2
        else:
            return 3
            
    def transform(self):
        self.X['bmi_in_hrange'] = self.X.apply(lambda x: self.bmi_healthy(x.Wt, x.Ht), axis=1)
        self.X['age_group'] = self.X.apply(lambda x: self.age_group(x.Ins_Age), axis=1)
        
        self.X.drop(columns=['Ht','Wt','IssueDate','Ins_Age'],inplace=True)
        return self.X
    
    def fit_transform(self,X, y=None):
        self.fit(X)
        return self.transform()


class InsuranceModel(RegressorMixin):
    def __init__(self):
        self.valid_features = ['AppID', 'Ins_Gender', 'bmi_in_hrange', 'age_group']
        
    def fit(self, X, y=None):
        pass
    
    def logic(self,AppID,gender,bmi_healthy,age_group):
        if bmi_healthy:
            base_value = 500
            reason = "BMI is in right range"
        else:
            if age_group == 0:
                base_value = 500
                reason = "BMI is in right range"
            elif age_group == 1:
                base_value = 750
                reason = "Age is between 18 to 39 and 'BMI' is either less than 17.49 or greater than 38.5"
            elif age_group == 2:
                base_value = 1000
                reason =  "Age is between 40 to 59 and 'BMI' is either less than 17.49 or greater than 38.5"
            else:
                base_value = 2000
                reason =  "Age isgreater than 60 and 'BMI' is either less than 17.49 or greater than 38.5"
        if gender == 'Female':
            return (AppID,base_value * 0.9, reason)
        else:
            return (AppID,base_value, reason)
        
        
        
    def predict(self,X):
        print(X.columns.to_list())
        for item in X.columns.to_list():
            if item not in self.valid_features:
                return 'Invalid Features'
        quote_values = X.apply(lambda x: self.logic(x.AppID,x.Ins_Gender,x.bmi_in_hrange,x.age_group), axis=1)
        return quote_values.to_list()
    
    def fit_predict(self,X, y=None):
        self.fit(X)
        return self.predict(X)
    
pipe = Pipeline([('transformer', TransformData()), ('InsuranceModel', InsuranceModel())])

