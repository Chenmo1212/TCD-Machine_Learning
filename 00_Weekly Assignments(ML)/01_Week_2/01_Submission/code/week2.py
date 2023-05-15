# Dataset Id：18-36--18

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt('week2.txt', delimiter=',')

# a.i
X1 = data[data[:, 2] == 1]
X2 = data[data[:, 2] == -1]

plt.scatter(X1[:, 0], X1[:, 1], marker='+', color='#06d6a0', label='+1')
plt.scatter(X2[:, 0], X2[:, 1], marker='o', color='#118ab2', s=10, label='-1')

plt.xlabel("X_1")
plt.ylabel("X_2")
plt.title("Figure 1: Scatter plot of data")
plt.legend()
plt.show()
print("""
===========================
a.i's answer

Ths answer is shown in Figure 1
=========================== \n
""")

# a.ii
X = data[:, :2]
Y = data[:, 2]
# Use the train:test to divide the dataset with a ratio of 2:8
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Separate the data with values of 1 and -1 in the train set to show different markers and colors in the same plot
positive = x_train[y_train == 1]
negative = x_train[y_train == -1]

plt.scatter(positive[:, 0], positive[:, 1], marker='+', color='#06d6a0', label='+1 train')
plt.scatter(negative[:, 0], negative[:, 1], marker='o', color='#118ab2', s=10, label='-1 train')
plt.xlabel("X_1")
plt.ylabel("X_2")

model = LogisticRegression(penalty='none', solver='lbfgs')
model.fit(x_train, y_train)

print("""
===========================
a.ii's answer

The fitting result of the model is:
""")
print('y = ' + str(model.coef_[0][0]) + ' * x1 ' + str(model.coef_[0][1]) + ' * x2 + ' + str(model.intercept_[0]))
print("=========================== \n")


# a.iii
print("""
===========================
a.iii's answer

Ths answer is shown in Figure 2
=========================== \n
""")
# Get the prediction results for the test set
y_pred = model.predict(x_test)

x_test_pos = x_test[y_pred == 1]
x_test_neg = x_test[y_pred == -1]
plt.scatter(x_test_pos[:, 0], x_test_pos[:, 1], marker='^', color='#ffd166', label='+1 test')
plt.scatter(x_test_neg[:, 0], x_test_neg[:, 1], marker='*', color='#ef476f', label='-1 test')


# Draw decision boundaries
def get_x2(x1, model):
    return (model.coef_[0][0] * x1 + model.intercept_) / - model.coef_[0][1]


dec_x1 = np.linspace(-1.0, 1.0, 100)
dec_x2 = get_x2(dec_x1, model)
plt.plot(dec_x1, dec_x2, color='#073b4c', label='decision boundaries')

plt.legend(loc=4)
plt.title("Figure 2: Plot of data and prediction result and decision boundary")
plt.show()

# b.i
print("""
===========================
b.i's answer

The fitting parameters for different C values are:
""")

from sklearn.svm import LinearSVC

# Set different penalty terms to get different model fitting results
model1 = LinearSVC(C=0.001).fit(x_train, y_train)
print('C=0.001', model1.intercept_, model1.coef_[0])

model2 = LinearSVC(C=1).fit(x_train, y_train)
print('C=1', model2.intercept_, model2.coef_[0])

model3 = LinearSVC(C=100, max_iter=10000).fit(x_train, y_train)
print('C=100', model3.intercept_, model3.coef_[0])

print("=========================== \n")

# b.ii
print("""
===========================
b.ii's answer

Ths answer is shown in Figure 3
=========================== \n
""")
# Substitute the test data set into different models to get different prediction results
pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)
pred3 = model3.predict(x_test)

# Displaying models with different C values in the same plot
x_pos_test = x_test[y_test == 1]
x_neg_test = x_test[y_test == -1]

fig = plt.figure(figsize=(12, 4))

# Draw decision boundaries
ax1 = fig.add_subplot(1, 3, 1)
dec_x1_1 = np.linspace(-1.0, 1.0, 100)
dec_x2_1 = get_x2(dec_x1_1, model1)
ax1.plot(dec_x1_1, dec_x2_1, color='#073b4c', label='Decision Boundary')

x_pos_1 = x_test[pred1 == 1]
x_neg_1 = x_test[pred1 == -1]
ax1.scatter(x_pos_1[:, 0], x_pos_1[:, 1], marker='^', color='#FFD166', label='+1 pred')
ax1.scatter(x_neg_1[:, 0], x_neg_1[:, 1], marker='*', color='#EF476F', label='-1 pred')
ax1.scatter(x_pos_test[:, 0], x_pos_test[:, 1], marker='+', color='#06D6A0', label='+1 actual')
ax1.scatter(x_neg_test[:, 0], x_neg_test[:, 1], marker='o', color='#118ab2', s=10, label='-1 actual')
ax1.legend(loc=4)

# Draw decision boundaries
ax2 = fig.add_subplot(1, 3, 2)
dec_x1_2 = np.linspace(-1.0, 1.0, 100)
dec_x2_2 = get_x2(dec_x1_2, model2)
ax2.plot(dec_x1_2, dec_x2_2, color='#073b4c', label='Decision Boundary')

x_pos_2 = x_test[pred2 == 1]
x_neg_2 = x_test[pred2 == -1]
ax2.scatter(x_pos_2[:, 0], x_pos_2[:, 1], marker='^', color='#FFD166', label='+1 pred')
ax2.scatter(x_neg_2[:, 0], x_neg_2[:, 1], marker='*', color='#EF476F', label='-1 pred')
ax2.scatter(x_pos_test[:, 0], x_pos_test[:, 1], marker='+', color='#06D6A0', label='+1 actual')
ax2.scatter(x_neg_test[:, 0], x_neg_test[:, 1], marker='o', color='#118ab2', s=10, label='-1 actual')
ax2.legend(loc=4)

# Draw decision boundaries
ax3 = fig.add_subplot(1, 3, 3)
dec_x1_3 = np.linspace(-1.0, 1.0, 100)
dec_x2_3 = get_x2(dec_x1_3, model3)
ax3.plot(dec_x1_3, dec_x2_3, color='#073b4c', label='Decision Boundary')

x_pos_3 = x_test[pred3 == 1]
x_neg_3 = x_test[pred3 == -1]
ax3.scatter(x_pos_3[:, 0], x_pos_3[:, 1], marker='^', color='#FFD166', label='+1 pred')
ax3.scatter(x_neg_3[:, 0], x_neg_3[:, 1], marker='*', color='#EF476F', label='-1 pred')
ax3.scatter(x_pos_test[:, 0], x_pos_test[:, 1], marker='+', color='#06D6A0', label='+1 actual')
ax3.scatter(x_neg_test[:, 0], x_neg_test[:, 1], marker='o', color='#118ab2', s=10, label='-1 actual')
ax3.legend(loc=4)
plt.figtext(0.5, 0.95, 'Figure 3: Plot of data and prediction result and decision boundary of different C', ha='center', va='top')
plt.show()

# b.iv
print("""
===========================
b.iv's answer

The accuracies for different C values are:
""")
from sklearn.metrics import accuracy_score

# Compare the Accuracy of Different Models
score = accuracy_score(y_pred, y_test)
score1 = accuracy_score(pred1, y_test)
score2 = accuracy_score(pred2, y_test)
score3 = accuracy_score(pred3, y_test)
print('Basic:     accuracy:', score)
print('C = 0.001: accuracy:', score1)
print('C = 1:     accuracy:', score2)
print('C = 100:   accuracy:', score3)
print("=========================== \n")

# c.i
print("""
===========================
c.i's answer

The fitting result of the model with the new features added is:
""")
x1 = x_train[:, 0]
x2 = x_train[:, 1]
# Insert a new feature column through the insert function
new_data = np.insert(x_train, 2, values=x1 * x1, axis=1)
new_data = np.insert(new_data, 3, values=x2 * x2, axis=1)

new_X = new_data
new_Y = y_train

# Refit the model with the new dataset
new_model = LogisticRegression(penalty='none', solver='lbfgs')
new_model.fit(new_X, new_Y)
print('y = ' + str(new_model.coef_[0][0]) + ' * x1 + ' + str(new_model.coef_[0][1]) + ' * x2 + ' + str(
    new_model.coef_[0][2]) + ' * x1 * x1 + ' + str(new_model.coef_[0][3]) + ' * x2 * x2 + ' + str(
    new_model.intercept_[0]))
print("=========================== \n")

# c.2
new_x_test = np.insert(x_test, 2, values=x_test[:, 0] ** 2, axis=1)
new_x_test = np.insert(new_x_test, 3, values=x_test[:, 1] ** 2, axis=1)

new_pred = new_model.predict(new_x_test)
x_new_pos = new_x_test[new_pred == 1]
x_new_neg = new_x_test[new_pred == -1]
plt.scatter(x_new_pos[:, 0], x_new_pos[:, 1], marker='^', color='#FFD166', label='+1 predict')
plt.scatter(x_new_neg[:, 0], x_new_neg[:, 1], marker='*', color='#EF476F', label='-1 predict')
plt.scatter(X1[:, 0], X1[:, 1], marker='+', color='#06D6A0', label='+1 actual')
plt.scatter(X2[:, 0], X2[:, 1], marker='o', color='#118ab2', s=10, label='-1 actual')
plt.xlabel("X_1")
plt.ylabel("X_2")
plt.legend(loc=4)
plt.title("Figure 4: Plot of data and prediction result of adding new features")
plt.show()

new_score = accuracy_score(new_pred, y_test)
print("""
===========================
c.ii's answer

Ths answer is shown in Figure 4

""")
print("The accuracy of the model is: %f" % new_score)
print("=========================== \n")

# c.3
# Create a baseline predictor using the results from the test data
base_pred = np.ones(y_test.shape)

model_score = accuracy_score(new_pred, y_test)
base_score = accuracy_score(base_pred, y_test)

print("""
===========================
c.iii's answer

""")
print("The accuracy of the baseline model is: %f" % base_score)
print("The accuracy of the final model is: %f" % model_score)

print("=========================== \n")


# c.iv
temp_x1 = np.linspace(new_data[:, 0].min(), new_data[:, 0].max(), 100)
temp_x2 = np.linspace(new_data[:, 1].min(), new_data[:, 1].max(), 100)
z = np.zeros((100, 100))

# Substitute each point into the model to calculate the predicted value,
# and store the result in z for displaying contour lines
for x_index, x in enumerate(temp_x1):
    for y_index, y in enumerate(temp_x2):
        temp = np.array([[x, y, x**2, y**2]])
        z[y_index][x_index] = new_model.predict(temp)

plt.scatter(X1[:, 0], X1[:, 1], marker='+', color='#06D6A0', label='+1 actual')
plt.scatter(X2[:, 0], X2[:, 1], marker='o', color='#118ab2', s=10, label='-1 actual')
plt.contour(temp_x1, temp_x2, z, colors='#073b4c')

plt.xlabel("X_1")
plt.ylabel("X_2")
plt.legend(loc=4)
plt.title("Figure 5: Plot of data and decision boundaries of model")
plt.show()
