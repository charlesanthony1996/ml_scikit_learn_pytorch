from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data[:, [2, 3]]
y = iris.target

print("Class labels ", np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)


print("Label counts in y: ", np.bincount(y))

print("Labels counts in y_train: ", np.bincount(y_train))

print("labels counts in y_test: ", np.bincount(y_test))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(x_train_std, y_train)

y_pred = ppn.predict(x_test_std)
print("Misclassified examples: %d" % (y_test != y_pred).sum())


from sklearn.metrics import accuracy_score
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y = x[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor='black')
    
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0], x_test[:, 1], c='none', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='Test set')

        
# x_combined_std = np.vstack((x_train_std, x_test_std))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(x= x_combined_std, y= y_combined, classifier=ppn, test_idx=range(105, 150))
# plt.xlabel("Petal length")
# plt.ylabel("Petal width")
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()




import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# z = np.arange(-7, 7, 0.1)
# sigma_z = sigmoid(z)
# plt.plot(z, sigma_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\sigma (z)$')
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()


# short code snippet to create a plot that illustrates the loss of classifying a single training example for different
# values of sigma(z)

def loss_1(z):
    return - np.log(sigmoid(z))

def loss_0(z):
    return - np.log(1 - sigmoid(z))

# z = np.arange(-10, 10, 0.1)
# sigma_z = sigmoid(z)
# c1 = [loss_1(x) for x in z]
# plt.plot(sigma_z, c1, label='L(w, b) if y=1')
# c0 = [loss_0(x) for x in z]
# plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')
# plt.ylim(0.0, 5.1)
# plt.xlim([0, 1])
# plt.xlabel('$\sigma(z)$')
# plt.ylabel('L(w, b)')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


# converting an adaline implementation into an algorithm for logistic regression

class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=1.0, size=x.shape[1])
        self.b_ = np.float_(0.)
        self.losses = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * x.T.dot(errors) / x.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1 - output))) / x.shape[0])
            self.losses_.append(loss)

        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    

    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)



x_train_01_subset = x_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[()]