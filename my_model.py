import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib as jb

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#%matplotlib inline

from info_apoio import fetch_housing_data
from info_apoio import load_housing_data
from info_apoio import CombinedAttributesAdder




# carregando os dados

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# criando diretorio

fetch_housing_data(housing_url=HOUSING_URL,
                   housing_path=HOUSING_PATH,
                   data="housing.tgz")

# lendo os dados

housing = load_housing_data(housing_path=HOUSING_PATH,
                            data="housing.csv")

housing['cut'] = pd.cut(housing['median_income'],
                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                        labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1,
                               test_size=0.2,
                               random_state=81)

for x, y in split.split(housing, housing['cut']):

    strat_train = housing.loc[x]
    strat_test = housing.loc[y]

for w in (strat_train, strat_test):
    w.drop('cut', axis=1, inplace=True)

ordinal_encoded = OrdinalEncoder()

imputer = SimpleImputer(strategy='median')

pipe = Pipeline([
                 ('imputer', SimpleImputer(strategy='median')),
                 ('attribs_adder', CombinedAttributesAdder()),
                 ('std_scaler', StandardScaler())              
                ])

housing = strat_train.copy()

my_model = pipe.fit_transform(housing.drop('ocean_proximity', axis=1))

jb.dump(my_model, 'my_model.pkl')
