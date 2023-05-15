import numpy as np
import sys
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc

data = np.loadtxt('../03_Code/week4_1.txt.bak', delimiter=',')
X = data[:, :2]
Y = data[:, 2]

X1 = data[data[:, 2] == 1]
X2 = data[data[:, 2] == -1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X1[:, 0], X1[:, 1], marker='+', color='#06d6a0', label='+1')
ax.scatter(X2[:, 0], X2[:, 1], marker='o', color='#118ab2', s=10, label='-1')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.legend(loc='best')
plt.title("Figure 1: Scatter plot of data")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def logi_tune_params(c_list, degree_list, x, y, cv=5):
    colors = ['#06d6a0', '#118ab2', '#ffd166', '#ef476f']
    for i, degree in enumerate(degree_list):
        x_poly = PolynomialFeatures(degree).fit_transform(x)
        accuracy = []
        accuracy_std = []
        for c in c_list:
            model = LogisticRegression(penalty='l2', C=c, solver='lbfgs', max_iter=1000)
            train_score = cross_val_score(model, x_poly, y, cv=cv, scoring='accuracy')
            model.fit(x_poly, y)

            accuracy.append(train_score.mean())
            accuracy_std.append(train_score.std())

        plt.errorbar(c_list, accuracy, yerr=accuracy_std, c=colors[i], label='degree = {}'.format(degree))
    plt.xlabel('Ci')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Relationships among Polynomial Degree, C and Accuracy")
    plt.show()


C_range = [1, 2, 3, 4, 5, 10, 15]
degree_range = [1, 2, 3, 4]
logi_tune_params(C_range, degree_range, x_train, y_train)


def knn_tune_params(k_list, x, y, cv=5):
    accuracy = []
    accuracy_std = []

    for i, k in enumerate(k_list):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_clf.fit(x, y)

        scores = cross_val_score(knn_clf, x, y, cv=cv, scoring='accuracy')
        accuracy.append(scores.mean())
        accuracy_std.append(scores.std())

    plt.errorbar(k_list, accuracy, yerr=accuracy_std, c='#06d6a0', label='accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Relationships between k and Accuracy")
    plt.show()


k_range = np.linspace(1, 20, 20, dtype=int)
knn_tune_params(k_range, x_train, y_train)

"""
混淆矩阵
"""

# logis
x_logis = PolynomialFeatures(2).fit_transform(x_train)
x_test_logis = PolynomialFeatures(2).fit_transform(x_test)

model_logis = LogisticRegression(penalty='l2', C=3)
model_logis.fit(x_logis, y_train)

y_pred_logis = model_logis.predict_proba(x_test_logis)
mat_logis = confusion_matrix(y_pred_logis, y_test)
print(mat_logis)

# Knn
# model_knn = KNeighborsClassifier(n_neighbors=18, weights='uniform')
# model_knn.fit(x_train, y_train)
# y_pred_knn = model_knn.predict_proba(x_test)
# mat_knn = confusion_matrix(y_pred_knn, y_test)
# print(mat_knn)

# baseline_pred = np.ones(y_test.shape)
# dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
# mat_baseline = confusion_matrix(baseline_pred, y_test)
# print(mat_baseline)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')

model = LogisticRegression()
fpr1, tpr1, thresholds_keras1 = roc_curve(y_test, y_pred_logis)
auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, label='Keras (area = {:.3f})'.format(auc1))

# model_knn = KNeighborsClassifier(n_neighbors=18, weights='uniform').fit(x_train, y_train)
# fpr2, tpr2, thresholds_keras2 = roc_curve(y_test, model_knn.decision_function(x_test))
# auc2 = auc(fpr2, tpr2)
# plt.plot(fpr2, tpr2, label='Keras (area = {:.3f})'.format(auc2))

# fpr3, tpr3, thresholds_keras3 = roc_curve(y_test, baseline_pred)
# auc3 = auc(fpr3, tpr3)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr3, tpr3, label='Keras (area = {:.3f})'.format(auc3))


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()