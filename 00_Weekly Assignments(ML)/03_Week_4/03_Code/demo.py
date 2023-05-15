# HG code
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error

df = pd.read_csv("week4_2.txt")
print(df.head())
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

plt.rc('font', size=18);
plt.rcParams['figure.constrained_layout.use'] = True


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
plt.plot(fpr, tpr, c='b', label='baseline most_frequent', linestyle='-.')

# dummy = DummyClassifier(strategy="uniform").fit(Xtrain, ytrain)
# fpr, tpr, _ = roc_curve(ytest, dummy.predict_proba(Xtest)[:, 1])
# plt.plot(fpr, tpr, c='y', label='baseline uniform', linestyle='--')
#
model = KNeighborsClassifier(n_neighbors=8, weights='uniform').fit(Xtrain, ytrain)
fpr, tpr, _ = roc_curve(ytest, model.predict_proba(Xtest)[:, 1])
plt.plot(fpr, tpr, c='r', label='kNN')

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(x_poly, y, test_size=0.2)
model = LogisticRegression(penalty='l2', C=6).fit(Xtrain, ytrain)
fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
plt.plot(fpr, tpr, c='g', label='Logistic Regression')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')
plt.show()