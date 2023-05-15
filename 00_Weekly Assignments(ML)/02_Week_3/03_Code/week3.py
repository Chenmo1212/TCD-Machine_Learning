# id:1--2--1

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

data = np.loadtxt('week3.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]

# i.a
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], label="data point")
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.set_zlabel('output')
plt.legend(loc='best')
plt.title("Figure 1: Scatter plot of data")
plt.show()

# 1.b
np.random.seed(333)
# Use the train:test to divide the dataset with a ratio of 2:8
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


class TrainModel:
    """
    Define a class that trains the results of the corresponding regression model
    based on the desired model type and corresponding parameters,
    and plots the corresponding graphs as required
    """
    model_class = None
    data = None
    degree = 1
    c_list = []
    model_list = []
    pred_list = []
    fig = None
    ax = None
    X1 = None
    X2 = None
    colors = ['#ef476f', '#118ab2', '#06d6a0', '#ffd166']

    def __init__(self, model_class, data, degree, c_list):
        self.model_class = model_class
        self.data = data
        self.degree = degree
        self.c_list = c_list
        self.get_model()

    def ModelRegression(self, alpha):
        return Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('reg', self.model_class(alpha=alpha))
        ])

    def get_coef(self, model):
        return model.named_steps['reg'].coef_

    def get_intercept(self, model):
        return model.named_steps['reg'].intercept_

    def get_model(self):
        self.model_list = []
        self.pred_list = []
        # Train different models by different values of C
        for c in self.c_list:
            reg = self.ModelRegression(1 / (2 * c))
            model = reg.fit(x_train, y_train)
            pred = reg.predict(x_test)

            self.model_list.append(model)
            self.pred_list.append(pred)

    def print_coefficients(self):
        for model in self.model_list:
            print(self.get_coef(model), self.get_intercept(model))

    def draw_origin(self, ax):
        """
        Get the maximum and minimum values of the original feature variables,
        expand 0.1 on this basis, and use them to display the plane
        """
        x1 = np.linspace(self.data[:, 0].min() - 0.1, self.data[:, 0].max() + 0.1)
        x2 = np.linspace(self.data[:, 1].min() - 0.1, self.data[:, 1].max() + 0.1)
        self.X1, self.X2 = np.meshgrid(x1, x2)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('predict')
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], label="data point")

    def draw_total(self):
        self.fig = plt.figure(figsize=(5, 5))
        ax = self.fig.add_subplot(111, projection='3d')
        self.draw_origin(ax)

        for i in range(1, len(self.model_list) + 1):
            self.draw_surface(ax, i)
        plt.legend()

        plt.title("Predictions for Different C")
        plt.show()

    def draw_single(self):
        self.fig = plt.figure(figsize=(10, 10))
        for i in range(1, len(self.model_list) + 1):
            ax = self.fig.add_subplot(2, 2, i, projection='3d')

            self.draw_origin(ax)
            self.draw_surface(ax, i)

            plt.legend()
            plt.title("Predictions for C = {}".format(self.c_list[i - 1]))
        plt.show()

    def draw_surface(self, ax, i):
        Z = []
        for j in range(len(self.X1)):
            temp = np.column_stack((self.X1[j], self.X1[j]))
            Z.append(self.model_list[i - 1].predict(temp))
        Z = np.array(Z)

        surf = ax.plot_surface(self.X1, self.X2, Z, label="C = {}".format(self.c_list[i - 1]),
                               color=self.colors[i - 1], alpha=0.4)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d


lasso_reg = TrainModel(model_class=Lasso, data=data, degree=5, c_list=[1, 10, 100, 1000])
lasso_reg.print_coefficients()
lasso_reg.draw_total()
lasso_reg.draw_single()

ridge_reg = TrainModel(model_class=Ridge, data=data, degree=5, c_list=[1, 10, 100, 1000])
ridge_reg.print_coefficients()
ridge_reg.draw_total()
ridge_reg.draw_single()


def draw_mean(model_class, c_list):
    fig = plt.figure(figsize=(10, 10))
    for i, c_range in enumerate(c_list):
        ax = fig.add_subplot(2, 2, i + 1)
        train_mean_error = []
        train_std_error = []
        test_mean_error = []
        test_std_error = []

        for c in c_range:
            model = model_class(alpha=1 / (2 * c))

            train_temp = []
            test_temp = []
            kf = KFold(n_splits=5)

            # Solve MSE for each categorical data
            for train, test in kf.split(X):
                model.fit(X[train], y[train])
                train_pred = model.predict(X[train])
                test_pred = model.predict(X[test])

                train_temp.append(mean_squared_error(y[train], train_pred))
                test_temp.append(mean_squared_error(y[test], test_pred))

            train_mean_error.append(np.array(train_temp).mean())
            train_std_error.append(np.array(train_temp).std())
            test_mean_error.append(np.array(test_temp).mean())
            test_std_error.append(np.array(test_temp).std())

        ax.errorbar(c_range, train_mean_error, yerr=train_std_error, c='#118ab2', label='train')
        ax.errorbar(c_range, test_mean_error, yerr=test_std_error, c='#ffd166', label='test')
        plt.xlabel('Ci')
        plt.ylabel('Mean square error')
        plt.legend()
        plt.title("{}: MSE for C = {}".format(i + 1, c_range))
    plt.show()


# ii.a
# Adjust different C values for training Lasso
C_range1 = [
    [1, 10, 100, 1000],
    [0.1, 1, 10, 100],
    [1, 5, 10, 15, 50],
    [1, 5, 10, 15, 20, 30]
]
draw_mean(Lasso, C_range1)

# ii.c

# Adjust different C values for training Ridge
C_range2 = [
    [1, 10, 100, 1000],
    [0.0001, 0.001, 0.1, 1, 10],
    [0.001, 0.01, 0.1, 1],
    [0.0001, 0.01, 0.1, 0.15, 0.2],
]
draw_mean(Ridge, C_range2)
