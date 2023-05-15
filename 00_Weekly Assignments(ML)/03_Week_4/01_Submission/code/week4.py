# id:25-50-25-1

import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.dummy import DummyClassifier

# load dataset
data = np.loadtxt('../03_Code/week4_2.txt', delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# show scatter of whole dataset
X1 = data[data[:, 2] == 1]
X2 = data[data[:, 2] == -1]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X1[:, 0], X1[:, 1], marker='+', color='#06d6a0', label='+1')
ax.scatter(X2[:, 0], X2[:, 1], marker='o', color='#118ab2', s=10, label='-1')
ax.set_xlabel('input x1')
ax.set_ylabel('input x2')
ax.legend(loc='best')
plt.title("Scatter plot of data")
plt.show()

# split dataset to train set and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=77)


# tuning logistic params
def logi_tune_params(c_list, degree_list, x, y, cv=5):
    colors = ['#06d6a0', '#118ab2', '#ffd166', '#ef476f']
    best_score = 0
    best_c = 0
    best_degree = 0

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

            # get the best k and score
            curr_score = train_score.mean()
            if curr_score > best_score:
                best_score = curr_score
                best_c = c
                best_degree = degree

        plt.errorbar(c_list, accuracy, yerr=accuracy_std, c=colors[i], label='degree = {}'.format(degree))
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Relationships among Polynomial Degree, C and Accuracy")
    plt.show()

    return best_c, best_degree, best_score


C_range = [1, 2, 3, 4, 5, 10, 15]
degree_range = [1, 2, 3, 4]
best_c_logic, best_degree, best_score_logic = logi_tune_params(C_range, degree_range, x_train, y_train)
print(best_c_logic, best_degree, best_score_logic)


# tuning knn params
def knn_tune_params(k_list, x, y, cv=5):
    accuracy = []
    accuracy_std = []
    best_score = 0
    best_k = 0

    for i, k in enumerate(k_list):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_clf.fit(x, y)

        scores = cross_val_score(knn_clf, x, y, cv=cv, scoring='accuracy')
        accuracy.append(scores.mean())
        accuracy_std.append(scores.std())

        # get the best k and score
        curr_score = scores.mean()
        if curr_score > best_score:
            best_score = curr_score
            best_k = k

    print(best_score, best_k)
    plt.errorbar(k_list, accuracy, yerr=accuracy_std, c='#06d6a0', label='accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title("Relationships between k and Accuracy")
    plt.show()

    return best_k, best_score


k_range = np.linspace(1, 30, 30, dtype=int)
best_k_knn, best_score_knn = knn_tune_params(k_range, x_train, y_train)

# calculate the matrix of logistic model, knn model and baseline model
# logistic
x_logis = PolynomialFeatures(best_degree).fit_transform(x_train)
x_test_logis = PolynomialFeatures(best_degree).fit_transform(x_test)

model_logis = LogisticRegression(penalty='l2', C=best_c_logic)
model_logis.fit(x_logis, y_train)

y_pred_logis = model_logis.predict(x_test_logis)
mat_logis = confusion_matrix(y_pred_logis, y_test)
print(mat_logis)

# Knn
model_knn = KNeighborsClassifier(n_neighbors=best_k_knn, weights='distance')
model_knn.fit(x_train, y_train)
y_pred_knn = model_knn.predict(x_test)
mat_knn = confusion_matrix(y_pred_knn, y_test)
print(mat_knn)

# Baseline
model_dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
y_pred_dummy = model_dummy.predict(x_test)
mat_dummy = confusion_matrix(y_test, y_pred_dummy)
print(mat_dummy)


def draw_matrix(confmat, title):
    text_list = ['TN', 'FP', 'FN', "TP"]

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    text_index = 0
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=str(confmat[i, j]) + str('({})').format(text_list[text_index]),
                    va='center', ha='center')
            text_index += 1
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title(title)


fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 3, 1)
draw_matrix(mat_logis, 'logistic')
ax = fig.add_subplot(1, 3, 2)
draw_matrix(mat_knn, 'knn')
ax = fig.add_subplot(1, 3, 3)
draw_matrix(mat_dummy, 'baseline')
plt.show()

# draw the roc curve of these models
x_logis = PolynomialFeatures(2).fit_transform(x_train)
x_test_logis = PolynomialFeatures(2).fit_transform(x_test)

model_logis = LogisticRegression(penalty='l2', C=5).fit(x_logis, y_train)
fpr1, tpr1, _ = roc_curve(y_test, model_logis.decision_function(x_test_logis))
roc_auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, label='logistic(auc = %0.4f)' % (roc_auc1))

model_knn = KNeighborsClassifier(n_neighbors=18, weights='uniform').fit(x_train, y_train)
fpr2, tpr2, _ = roc_curve(y_test, model_knn.predict_proba(x_test)[:, 1])
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, label='knn(auc = %0.4f)' % (roc_auc2))

dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
fpr3, tpr3, _ = roc_curve(y_test, dummy.predict_proba(x_test)[:, 1])
roc_auc3 = auc(fpr3, tpr3)
plt.plot(fpr3, tpr3, c='y', label='baseline(auc = %0.4f)' % (roc_auc3), linestyle='--')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
