import numpy as np
import pandas as pd

#load data
df = pd.read_csv('titanic_data.csv')
#limit to categorical values
df = df.select_dtypes(include=[object])

print(df.head())


#apply lable encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feature in df:
    df[feature] = le.fit_transform(df[feature])

print(df.head())

#apply one hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

dft = ohe.fit_transform(df)
print(ohe.n_values_)
print(type(dft))
print(dft.shape)
