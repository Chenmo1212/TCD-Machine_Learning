import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import TargetEncoder
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

random_state = 23


def nanfill(data, row, column):
    tag = data.iloc[row, 1]
    # print(data[data.iloc[:,1] == tag].iloc[:,column],'\n')
    temp = data[data.iloc[:, 1] == tag].iloc[:, column]
    temp = np.average(temp[temp.notna()])
    if (pd.isna(temp)):
        print(row, column)
    data.iloc[row, column] = temp


def normalization(data):
    for i in range(data.shape[1]):
        avg = data.iloc[:, i].mean()
        std = data.iloc[:, i].std(ddof=1)
        data.iloc[:, i] = (data.iloc[:, i] - avg) / std


# Calculate the mean and standard deviation of the cross-validation results,
# the default is five-fold cross-validation, and the result is the correlation coefficient R2
def cross_validate(models, cv=5, scoring='r2'):
    means = []
    stds = []
    best_score = 0
    best_index = 0

    for index, model in enumerate(models):
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        means.append(score.mean())
        stds.append(score.std(ddof=1))

        # get the best alpha and score
        curr_score = score.mean()
        if curr_score > best_score:
            best_score = curr_score
            best_index = index
    return means, stds, best_index


def MAE(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    mae_train = mean_absolute_error(ytrain, ypred)
    ypred = model.predict(Xtest)
    mae_test = mean_absolute_error(ytest, ypred)
    print("MAE    Train:%f      Test:%f" % (mae_train, mae_test))
    # print("The MAE for the %s model with the training data was %f and with the test data was %f"%(name, mae_train, mae_test))


def MSE(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    mse_train = mean_squared_error(ytrain, ypred, squared=True)
    ypred = model.predict(Xtest)
    mse_test = mean_squared_error(ytest, ypred, squared=True)
    print("MSE    Train:%f      Test:%f" % (mse_train, mse_test))
    # print("The RMSE for the %s model with the training data was %f and with the test data was %f" % (name, mse_train, mse_test))


def r_sq(name, model, Xtrain, Xtest, ytrain, ytest):
    ypred = model.predict(Xtrain)
    rsq_train = r2_score(ytrain, ypred)
    ypred = model.predict(Xtest)
    rsq_test = r2_score(ytest, ypred)
    print("RSQ    Train:%f      Test:%f" % (rsq_train, rsq_test))
    # print("The R squared score for the %s model with the training data was %f and with the test data was %f" % (name, rsq_train, rsq_test))


def plot(name, model, Xtrain, Xtest, ytrain, ytest, filename):
    ypred = model.predict(Xtrain)
    fig, axs = plt.subplots()
    axs.scatter(Xtrain[:, 1], ytrain, s=20, c='b', marker='+')
    axs.scatter(Xtrain[:, 1], ypred, s=20, c='r', marker='+')
    axs.set_title(name + ' - Precicted Training Data')
    axs.set_xlabel('feature')
    axs.set_ylabel('amount')
    axs.legend(['Target', 'Predictions'])
    fig.show()
    # fig.savefig('Plots/Prediction Plots/' + filename + '_train_pred')

    ypred = model.predict(Xtest)
    fig, axs = plt.subplots()
    axs.scatter(Xtest[:, 1], ytest, s=20, c='b', marker='+')
    axs.scatter(Xtest[:, 1], ypred, s=20, c='r', marker='+')
    axs.set_title(name + ' - Predicted Test Data')
    axs.set_xlabel('feature')
    axs.set_ylabel('amount')
    axs.legend(['Target', 'Predictions'])
    fig.show()


def plot_model_res(model, name):
    strategy = "mean"
    print("{}:".format(name))
    MAE("{}".format(name), model, X_train, X_test, y_train, y_test)
    MSE("{}".format(name), model, X_train, X_test, y_train, y_test)
    r_sq("{}".format(name), model, X_train, X_test, y_train, y_test)
    plot("{} Regression".format(name), model, X_train, X_test, y_train, y_test, "d")


def correlate():
    group = [['LACCESS_POP15', 'LACCESS_LOWI15', 'LACCESS_HHNV15', 'LACCESS_CHILD15', 'LACCESS_SENIORS15'],
             ['GROCPTH16', 'SUPERCPTH16', 'CONVSPTH16', 'SPECSPTH16', 'WICSPTH16'],
             ['FFRPTH16', 'FSRPTH16'],
             ['FOODINSEC_15_17', 'VLFOODSEC_15_17'],
             ['FMRKT_WIC18', 'FMRKT_WICCASH18'],
             ['POVRATE15', 'CHILDPOVRATE15']]

    for i in range(len(group)):
        corr = raw_data[group[i]].corr()
        plt.figure(figsize=(12, 8), dpi=300)
        #     sns.set(font_scale=2)
        sns.heatmap(corr, linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
        plt.savefig('corr_heatmap_{}.jpg'.format(i))


def clean_data(raw_data):
    preserve_columns = ['State', 'LACCESS_POP15', 'GROCPTH16',
                        'SUPERCPTH16', 'CONVSPTH16', 'SPECSPTH16', 'WICSPTH16', 'FFRPTH16', 'FSRPTH16',
                        'FOODINSEC_15_17', 'FMRKT_WIC18', 'POVRATE15', 'PCT_WIC17']

    raw_data['PCT_WIC17'] = raw_data['PCT_WIC17'] * raw_data['Population_Estimate_2016'] / 100
    raw_data['FOODINSEC_15_17'] = raw_data['FOODINSEC_15_17'] * raw_data['Population_Estimate_2016'] / 100
    raw_data['POVRATE15'] = raw_data['POVRATE15'] * raw_data['Population_Estimate_2016'] / 100

    tmp_data = raw_data[preserve_columns]
    temp = tmp_data.iloc[:, 1:]

    normalization(temp)
    raw_data.iloc[:, 1:] = temp
    enc = TargetEncoder(cols=['State']).fit(raw_data['State'], raw_data['PCT_WIC17'])
    raw_data['State'] = enc.transform(raw_data['State'])

    return raw_data.sample(frac=1, random_state=random_state).values


def baseline():
    dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
    dummy_score = cross_val_score(dummy, X_train, y_train, cv=5, scoring='r2')

    print(dummy_score.mean(), dummy_score.std())
    print(dummy_score)

    model_dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)

    plot_model_res(model_dummy, 'Dummy')


def Elastic():
    alphas = [0.1, 0.01, 0.005, 0.001]
    l1_ratio = [0.25, 0.5, 0.75, 0.9]

    best_score = 0
    best_alpha = 0
    best_ratio = 0

    fig = plt.figure(figsize=(12, 8))

    for i, alpha in enumerate(alphas):
        mean_list = []
        std_list = []
        ElasticNet_clfs = []

        for radio in l1_ratio:
            model = ElasticNet(alpha=alpha, l1_ratio=radio, random_state=1212, max_iter=5000).fit(X_train, y_train)
            ElasticNet_clfs.append(model)

        mean_tmp, std_tmp, best_index = cross_validate(ElasticNet_clfs)

        if best_score < mean_tmp[best_index]:
            best_score = mean_tmp[best_index]
            best_ratio = l1_ratio[best_index]
            best_alpha = alpha

        ax = fig.add_subplot(2, 2, i + 1)
        plt.xlabel('L1_ratio')
        plt.ylabel('R² - Score')
        ax.errorbar(l1_ratio, mean_tmp, yerr=std_tmp, c='#06d6a0', label="alhpa = {}".format(alpha))
        plt.legend()

    fig.suptitle("Errorbar Plot of different Alpha and L1_ratio", fontsize=16)
    plt.show()

    print('score', best_score)
    print('alpha', best_alpha)
    print('ratio', best_ratio)

    model_EN = ElasticNet(alpha=best_alpha, l1_ratio=best_ratio, random_state=random_state, max_iter=5000).fit(X_train,
                                                                                                               y_train)
    plot_model_res(model_EN, 'Elastic Net')


def Decision():
    DecisionTrees = []
    depth_list = [3, 5, 7, 11, 13, 15]

    for depth in depth_list:
        clf = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        DecisionTrees.append(clf)

    mean_dt, std_dt, index = cross_validate(DecisionTrees)

    best_score = mean_dt[index]
    best_depth = depth_list[index]

    print(best_score, best_depth)

    fig, axs = plt.subplots()
    axs.errorbar(depth_list, mean_dt, yerr=std_dt, c='#06d6a0')
    axs.set_title("Errorbar plot of different max_depth")
    axs.set_xlabel('max_depth')
    axs.set_ylabel('R² - Score')
    plt.show()

    model_dt = DecisionTreeRegressor(max_depth=best_depth, random_state=random_state).fit(X_train, y_train)
    plot_model_res(model_dt, 'dt')


def MLP():
    NNs = []
    size_list = [(64, 8), (64, 16, 4), (128, 16), (128, 32, 8)]
    alpha_list = [0.1, 0.01, 0.001, 0.0001]

    best_score = 0
    best_size = ()
    best_alpha = 0

    for i in range(len(size_list)):
        NN_same_size = []
        for alpha in alpha_list:
            clf = MLPRegressor(hidden_layer_sizes=size_list[i],
                               learning_rate_init=alpha,
                               random_state=random_state, max_iter=2000)
            NN_same_size.append(clf)
        NNs.append(NN_same_size)

    NN_means = []
    NN_stds = []
    for i in range(len(NNs)):
        NN_mean, NN_std, index = cross_validate(NNs[i])
        NN_means.append(NN_mean)
        NN_std.append(NN_std)

        curr_score = NN_mean[index]
        if curr_score > best_score:
            best_score = curr_score
            best_size = size_list[index]
            best_alpha = alpha_list[index]

    print(best_score, best_size, best_alpha)

    model_mlp = MLPRegressor(hidden_layer_sizes=best_size,
                             learning_rate_init=best_alpha,
                             random_state=random_state, max_iter=2000).fit(X_train, y_train)

    plot_model_res(model_mlp, 'mlp')


if __name__ == '__main__':
    # import data
    raw_data = pd.read_csv('Data.csv')

    for i in range(3):
        for j in range(raw_data.shape[0]):
            raw_data.iloc[j, i + 2] = np.float64(raw_data.iloc[j, i + 2].replace(',', ''))

    for i in range(raw_data.shape[0]):
        for j in range(raw_data.shape[1]):
            if (pd.isna(raw_data.iloc[i, j])):
                nanfill(raw_data, i, j)

    correlate()  # correlation analysis

    data_clean = clean_data(raw_data)  # clean data
    X_train = np.float64(data_clean[:2512, 1:-1])
    y_train = np.float64(data_clean[:2512, -1])
    X_test = np.float64(data_clean[2512:, 1:-1])
    y_test = np.float64(data_clean[2512:, -1])

    baseline()  # baseline analysis

    Elastic()  # elastic

    Decision()  # decision

    MLP()  # MLP
