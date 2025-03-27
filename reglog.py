import matplotlib
import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions

matplotlib.use('TkAgg')


class MultiRegressionGD:
    def __init__(self, eta=0.05, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.W = None

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_classes = np.unique(y).shape[0]

        self.W = rgen.normal(loc=0.0, scale=0.01, size=(n_features + 1, n_classes))

        X_bias = np.c_[np.ones((n_samples, 1)), X]

        for _ in range(self.n_iter):
            Z = X_bias.dot(self.W)
            probabilities = self.softmax(Z)
            y_one_hot = np.eye(n_classes)[y]

            gradient = X_bias.T.dot(y_one_hot - probabilities) / n_samples
            self.W += self.eta * gradient

    def predict_proba(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return self.softmax(X_bias.dot(self.W))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model = MultiRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    def format_probs(probs):
        return {k: round(float(v), 4) for k, v in enumerate(probs)}

    for i, probs in enumerate(model.predict_proba(X_test)):
        print(f"Obj {i}: {format_probs(probs)}")

    plot_decision_regions(X=X_train, y=y_train, classifier=model)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
