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

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Non flavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']

print('class labels', np.unique(df_wine['Class label']))

print(df_wine.head())

from sklearn.model_selection import train_test_split

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
x_train , x_test, y_train , y_test = train_test_split(x, y, test_size=0.3,random_state=0, stratify=y)

# min max scaling procedure

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
x_train_norm = mms.fit_transform(x_train)
x_test_norm = mms.transform(x_test)

# examples of standardization and normalization
ex = np.array([0, 1, 2, 3, 4, 5])
print("standardized: ", (ex - ex.mean()) / ex.std())
print("normalized: ", (ex - ex.min()) / (ex.max() - ex.min()))


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()

x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')

lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(x_train_std, y_train)

print("Training accuracy: ", lr.score(x_train_std, y_train))
print("Testing accuracy: ", lr.score(x_test_std, y_test))

# intercept

print("Intercept: ", lr.intercept_)
print("coefficient:", lr.coef_)

# regularization strength example

import matplotlib.pyplot as plt
fig = plt.figure()

ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue',
           'gray', 'indigo', 'orange']
weights, params = [], []

for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)

    lr.fit(x_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+ 1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Weight coefficient')
plt.xlabel('C (inverse regularization strength)')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
# plt.show()


# sequential feature selection algorithms

from sklearn.base import clone
from itertools import combinations 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state= 1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        dim = x_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(x_train, y_train, x_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r= dim - 1):
                score = self._calc_score(x_train, y_train, score.append(score), subsets.append(p))
                best = np.argmax(scores)
                self.indices_ = subsets[best]
                self.subsets_.append(self.indices_)
                dim -= 1

            self.scores_.append(scores[best])
        
        self.k_score_ = self.scores_[-1]

        return self
    

    def transform(self, x):
        return x[:, self.indices_]
    
    def _calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score



