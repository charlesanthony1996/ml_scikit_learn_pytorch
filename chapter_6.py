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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=1)
print(x_train.shape)
print(len(x_train))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression())
pipe_lr.fit(x_train, y_train)
y_pred = pipe_lr.predict(x_test)
test_acc = pipe_lr.score(x_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")


# stratified k fold validation

import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10).split(x_train, y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(x_train[train], y_train[train])
    score = pipe_lr.score(x_train[test], y_train[test])
    scores.append(score)
    print(f"k fold: {k+1:02d}, ", f"Class distr.: {np.bincount(y_train[train])}, ", f"Acc.: {score:.3f}")
    
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f"\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = pipe_lr, X= x_train, y=y_train, cv= 10, n_jobs= 1)
print("Accuracy scores: {scores}")

print(f"CV accuracy: {np.mean(scores):.3f} " f"+/- {np.std(scores):.3f}")
