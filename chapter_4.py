# building good training datasets - data preprocessing

# identifying missing values in tabular data

import pandas as pd
from io import StringIO


csv_data = \
'''
A, B, C, D
1.0, 2.0, 3.0, 4.0
5.0, 6.0,, 8.0
10.0, 11.0, 12.0,
'''

df = pd.read_csv(StringIO(csv_data))
print(df.head())

print(df.isnull().sum())

# eliminating training examples or features with missing values

print(df.dropna(axis=0))

print(df.dropna(axis=1))

print(df.dropna(how='all'))


# inputting missing values

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)

print(df.fillna(df.mean()))

# categorical data encoding with pandas

import pandas as pd
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])

df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

# mapping ordinal features

size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df)

inv_size_mapping = { v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

import numpy as np

class_mapping = { label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

inv_class_mapping = { v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

print(df)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
print(y)

class_le.inverse_transform(y)
print(class_le.inverse_transform(y))

# performing one hot encoding on nominal features

x = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
x[:, 0] = color_le.fit_transform(x[:, 0])
print(x)

from sklearn.preprocessing import OneHotEncoder
x = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(x[:, 0].reshape(-1, 1)).toarray()


from sklearn.compose import ColumnTransformer

x = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
# print(c_transf)
c_transf.fit_transform(x).astype(float)
print(c_transf.fit_transform(x).astype(float))

pd.get_dummies(df[['price', 'color', 'size']])
print(pd.get_dummies(df[['price', 'color', 'size']]))


pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
# blue was dropped here
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))


color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1, 2])
])

c_transf.fit_transform(x).astype(float)
print(c_transf.fit_transform(x).astype(float))

# optional: encoding ordinal features

df = pd.DataFrame([['green', 'M', 10.1, 'class2'], ['red', 'L', 13.5, 'class1'], ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

df['x > M'] = df['size'].apply(lambda x: 1 if x in { 'L', 'XL'} else 0)

df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)

del df['size']
print(df)

# paritioning a dataset into seperate training and test datasets

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# print(df_wine.head())

df.wine_columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Non flavanoid pehnols', 'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']