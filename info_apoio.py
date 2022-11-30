import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedShuffleSplit




def fetch_housing_data(housing_url, housing_path, data):
     
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, data)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
    
    
def load_housing_data(housing_path, data):
    csv_path = os.path.join(housing_path, data)
    return pd.read_csv(csv_path)

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    '''Classe responsavel por add atributos'''
            
    def __init__(self, add_badrooms_per_room=True):
        
        self.add_badrooms_per_room = add_badrooms_per_room
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        
        #rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        
        room_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_badrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, room_per_household, population_per_household]
        
        
        
        
        
def data_preparation(df, cat):
    split = StratifiedShuffleSplit(n_splits=1,
                                test_size=0.2,
                                random_state=81)

    for x, y in split.split(df, df[cat]):
        strat_train = df.loc[x]
        strat_test = df.loc[y]

    for w in (strat_train, strat_test):
        w.drop(cat, axis=1, inplace=True)
        
        return strat_train, strat_test        

    

def data_trasformation(df, cat_att):
    
    pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('attribs_adder', CombinedAttributesAdder()),
                    ('std_scaler', StandardScaler())])             
                    
    num_att = list(df.drop(cat_att, axis=1))

    full_pipeline = ColumnTransformer([
        ('num', pipe, num_att),
        ('cat', OneHotEncoder(), [cat_att])
        ])
    
    
    
    return full_pipeline    
    
    
def fit_and_evaluate(model):
    
    model.fit(X_train, y_train)
    
    model_pred = model.predict(X_test)
    model_mae = np.mean(abs(y_test - model_pred))
    
    return model_mae    
        
         