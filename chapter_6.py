import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
# print(df.head())

from sklearn.preprocessing import LabelEncoder
x = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)

print(le.transform(['M', 'B']))


from sklearn.model_selection import train_test_split

x_train, x_train, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=1)
print(x_train.shape)
print(len(x_train))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.

