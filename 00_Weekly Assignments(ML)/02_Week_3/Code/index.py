import numpy as np

X = np.arange(0, 1, 0.05).reshape(-1, 1)
y = 10 * X + np.random.normal(0.0, 1.0, X.size).reshape(-1, 1)

mean_error = []
std_error = []
Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100]
for Ci in Ci_range:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1 / (2 * Ci))
    temp = []
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        from sklearn.metrics import mean_squared_error

        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

import matplotlib.pyplot as plt

plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('Ci')
plt.ylabel('Mean square error')
plt.xlim((0, 50))
plt.show()
