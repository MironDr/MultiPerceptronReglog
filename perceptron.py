import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions

matplotlib.use('TkAgg')


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class MultiClassPerceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.classes_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for class_ in self.classes_:
            y_bin = np.where(y == class_, 1, -1)
            ppn = Perceptron(eta=self.eta, n_iter=self.n_iter)
            ppn.fit(X, y_bin)
            self.classifiers.append(ppn)

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        return self.classes_[np.argmax(predictions, axis=0)]


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    mc_ppn = MultiClassPerceptron(eta=0.1, n_iter=1000)
    mc_ppn.fit(X_train, y_train)

    plot_decision_regions(X=X_train, y=y_train, classifier=mc_ppn)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
