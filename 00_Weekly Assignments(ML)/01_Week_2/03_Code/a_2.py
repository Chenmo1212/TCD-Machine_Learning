import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y < 2, :2]  # 取标签为0,1的数据，取特征的前两个，
y = y[y < 2]  # 取标签0,1


def plot_decision_boundary(model, axis):  # 定义决策边界绘制函数
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    print(X_new.shape)
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


def PolynomialLogisticRegression(degree):  # 定义多项式logistics回归
    return Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("std_scaler", StandardScaler()),
        ("log_reg", LogisticRegression())
    ])


poly_log_reg = PolynomialLogisticRegression(degree=2)  # 建立二次多项式logistics回归实例
poly_log_reg.fit(X, y)
poly_log_reg.score(X, y)  # 输出0.97

plot_decision_boundary(poly_log_reg, axis=[2, 8, 1, 6])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
