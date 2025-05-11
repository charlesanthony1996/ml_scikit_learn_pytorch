# compressing data via dimensionality reduction

# extracting the principal components step by step

import pandas as pd
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

print(df_wine.head())

from sklearn.model_selection import train_test_split

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

# print(x_train_std)
# print(x_test_std)

import numpy as np
cov_mat = np.cov(x_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigen Values: ", eigen_vals)


# cumulative sum of explained variances

tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.figure()
plt.bar(range(1, 14), var_exp, align='center', label='Individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.show()

# sorting the eigenpairs by decreasing order of the eigenvalues

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print("Matrix W:\n", w)

x_train_std[0].dot(w)

x_train_pca = x_train_std.dot(w)

# two dimensional scatterplot

colors = ['r', 'b', 'g']
markers = ['o', 's', '^']
plt.figure()
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_pca[y_train == l, 0], x_train_pca[y_train == l, 1], c=c, label=f"Class {l}", marker=m)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc='lower left')
plt.tight_layout()
# plt.show()


# principal component analysis in scikit learn

from matplotlib.colors import ListedColormap
def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.ylim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    # plot class examples
    # plt.figure()
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x= x[y == cl, 0], y= x[y == cl, 1], alpha=0.8, c= colors[idx], marker=markers[idx], 
                    label=f"Class {cl}", edgecolor='black')
    # plt.show()

plt.figure()
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


pca = PCA(n_components= 2)
lr = LogisticRegression(multi_class='ovr', random_state= 1, solver='lbfgs')
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)

lr.fit(x_train_pca, y_train)
plot_decision_regions(x_train_pca, y_train, classifier=lr)

plt.xlabel('pc 1')
plt.ylabel('pc 2')

plt.legend(loc='lower left')
plt.tight_layout()
# plt.show()

plt.figure()
plot_decision_regions(x_test_pca, y_test, classifier=lr)
plt.xlabel('pc 1')
plt.ylabel('pc 2')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.show()


pca = PCA(n_components=None)
x_train_pca = pca.fit_transform(x_train_std)
pca.explained_variance_ratio_

# assessing feature contributions

loadings = eigen_vecs * np.sqrt(eigen_vals)

fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('loadings for pc 1')
ax.set_xticklabels(df_wine.columns[1:], rotation=90)
plt.figure()
plt.ylim([-1, 1])
plt.tight_layout()
# plt.show()


# computing the scatter matrices

np.set_printoptions(precision= 1)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(x_train_std[y_train == label,], axis=0))
    print(f"MV {label}: {mean_vecs[label - 1]}\n")


d = 13
s_w = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in x_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row  - mv).T)
    s_w += class_scatter

print("Within class scatter matrix: ", f"{s_w.shape[0]} x {s_w.shape[1]}")


print("Class label distribution: ", np.bincount(y_train)[1:])

# code for computing the scaled within class scatter matrix

d = 13
s_w = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(x_train_std[y_train == label].T)
    s_w += class_scatter

print("Scaled within class scatter matrix: ", f"{s_w.shape[0]}x{s_w.shape[1]}")


