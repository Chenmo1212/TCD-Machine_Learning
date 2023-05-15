# Project Proposal

## motivation

After discussion among the group members, we finally decided to put emphasis on the relation between a variety of factors and the actual number of participants in the WIC program(WIC is an American federal assistance program for healthcare and nutrition of low-income pregnant women, breastfeeding women, and children under the age of five). 


The eligibility of participants is made up of various factors, and the number of participants plays a very important role in this program. Therefore, we will build a model based on the dataset "Food Environment Atlas" to analyse and evaluate which factors influence the amount of participants in the U.S. WIC program and which factors significantly outperform others. Through this process, we hope to analyse and forecast the number of participants in future WIC programs, thereby improving their budget planning.


## Dataset

We will use the Food Environment Atlas dataset from data.gov(see the link below). The dataset contains a variety of features that can be fed to the training process such as many food environment factors such as proximity of stores/restaurants, food prices, food assistance programs, community characteristics, and also includes the population size of each city (county).


## Method

We want to evaluate the most frequently used supervised machine learning algorithms such as linear regression, kernelized SVM,  kernelized ridge regression etc until we decide which one outperforms the others significantly. Before implementing the algorithms, we need to choose the features correlated with the number of participants of WIC via evaluating different combinations of original features and/or polynomial transformation. For each model, we use cross-validation to select hyperparameters then select the best model. Finally, We will use MSE to evaluate machine learning algorithms.

## Intended Experiments

We decide to evaluate the performance of each model, see how accurate they are when predicting the future sale prices. Due to the sufficiency of the data, we can split the dataset into two sets: one for training and one for testing where the split ratio is 80%:20%, and split the train set to train set and validation set based on cross-validation. After we produce the model and determine the hyperparameter, we will firstly compare it to a baseline model.  


Reference link:
https://catalog.data.gov/dataset/food-environment-atlas

