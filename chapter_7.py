# learning with ensembles

from scipy.special import comb
import math
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * 
            error ** k * 
            (1- error) ** (n_classifier - k)
            for k in range(k_start, n_classifier + 1)]
    return sum(probs)

ensemble_error(n_classifier=11, error=0.25)
print(ensemble_error(n_classifier=11, error=0.25))


import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range, ens_errors, label='Ensemble error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--', label='Base error', linewidth=2)

plt.xlabel('Base error')
plt.ylabel('Base / Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
# plt.show()

# implementing a simple majority vote classifier

import numpy as np
np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))
# print(np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6])))

ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p)

np.argmax(p)
# print(np.argmax(p))

# putting everything together now

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class majorityvoterclassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key,
            value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, x, y):
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"'vote must be probability'"
                             f"or 'classlabel'"
                             f'; got {len(self.weights)} weights, '
                             f' {len(self.classifiers)} classifiers')
        
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(x, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    

    def predict(self, x):
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(x), axis= 1)
        else:
            predictions = np.asarray([
                clf.predict(x) for clf in self.classifiers_
            ]).T

            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self.weights)
                ),
                axis = 1, arr=predictions
            )
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, x):
        probas = np.asarray([clf.predict_proba(x) for clf in self.classfiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.params(deep=True).items():
                    out[f'{name}___{key}'] = value
            return out
        





# using the majority voting principle to make predictions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


iris = datasets.load_iris()
