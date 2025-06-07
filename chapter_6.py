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


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=10000))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=x_train, y=y_train, train_sizes=np.linspace(0.1, 1, 10), 
                                                       cv=10, 
                                                       n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label="Validation accuracy")

plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
# plt.show()

# addressing over and underfitting with validation curves

from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 10, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_lr, X =x_train, y=y_train, param_name="logisticregression__C",
                                             param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.figure()
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = 0.15, color='blue')

plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
# plt.show()

# tuning hyperparameters via grid search

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state= 1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},{'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, refit=True, n_jobs=-1)
gs = gs.fit(x_train, y_train)
print(gs.best_score_)

print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(x_train, y_train)
print(f'Test accuracy: , {clf.score(x_test, y_test):.3f}')

# exploring hyperparameter configurations more widely with randomized search

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

import scipy
param_range = scipy.stats.loguniform(0.0001, 1000.0)

np.random.seed(1)
param_range.rvs(10)


from sklearn.model_selection import RandomizedSearchCV

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, {'svc__C': param_range, 
                                                                   'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

rs = RandomizedSearchCV(
    estimator=pipe_svc, 
    param_distributions=param_grid,
    scoring='accuracy',
    refit=True,
    n_iter=20,
    cv=10,
    random_state=1,
    n_jobs=-1
)


rs = rs.fit(x_train, y_train)
print(rs.best_score_)

print(rs.best_params_)


# more resource efficient hyperparameter search with successive halving

from sklearn.experimental import enable_halving_search_cv


from sklearn.model_selection import HalvingRandomSearchCV

hs = HalvingRandomSearchCV(
    pipe_svc, 
    param_distributions=param_grid, 
    n_candidates='exhaust', 
    resource='n_samples', 
    factor=1.5, 
    random_state=1,
    n_jobs=1
)

hs = hs.fit(x_train, y_train)
print(hs.best_score_)

clf = hs.best_estimator_
print(f"Test accuracy: {hs.score(x_test, y_test):.3f}")

# alogrithm selection with nested cross validation

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']}, {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=2)

scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)

print(f"cv accuracy: {np.mean(scores):.3f} "  f'+/- {np.std(scores):.3f}')



from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
    scoring='accuracy',
    cv=2
)

scores = cross_val_score(gs, x_train, y_train, scoring='accuracy', cv=5)
print(f'CV accuracy: {np.mean(scores):.3f}' f'+/- {np.std(scores):.3f}')

# reading a confusion matrix

from sklearn.metrics import confusion_matrix
pipe_svc.fit(x_train, y_train)
y_pred = pipe_svc.predict(x_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x= j, y=i, s=confmat[i, j], va='center', ha='center')

ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label')
plt.ylabel('True label')
# plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import matthews_corrcoef

pre_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Precision: {pre_val:.3f}')

rec_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f'Recall: {rec_val:.3f}')

f1_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f"F1 val: {f1_val:.3f}")

mcc_val = precision_score(y_true=y_test, y_pred=y_pred)
print(f"MCC val: {mcc_val:.3f}")

from sklearn.metrics import make_scorer
c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{'svc__C': c_gamma_range,
               'svc__kernel': ['linear']},
               {'svc__C': c_gamma_range,
                'svc__gamma': c_gamma_range,
                'svc__kernel': ['rbf']}]

scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)



# plotting a receiver operating characteristic

from sklearn.metrics import roc_curve, auc

from numpy import interp
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', C= 100.0)
)

x_train2 = x_train[:, [4, 14]]
# print(x_train2)

cv = list(StratifiedKFold(n_splits=3).split(x_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 1000)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(
        x_train2[train],
        y_train[train]
    ).predict_proba(x_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC fold {i+1} (area= {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing {}')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (area = {mean_auc:.2f})', lw=2)

plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance (area=1.0)')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
# plt.show()


pre_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')


# dealing with class imbalance
x_imb = np.vstack((x[y == 0], x[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100

