# import data tools
import numpy as np
import pandas as pd
# import visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
# import modeling evaluation tools
from sklearn.metrics import mean_squared_error, explained_variance_score
# import linear regression models
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
# import feature selection tools
from sklearn.feature_selection import SelectKBest, f_regression, RFE
# import clustering models
from sklearn.cluster import KMeans

# -------------------------------------------------------------

def get_baseline_model(y_train):
    '''
    This function will print the mean baseline RMSE and R^2 scores
    '''
    # get the mean value of quality scores from the training data
    y_train['quality_pred_mean'] = y_train.quality.mean()
    # calculate RMSE for the mean
    rmse_train_mu = mean_squared_error(y_train.quality,
                                   y_train.quality_pred_mean, squared=False)
    # print results
    print('Baseline Model (mean)')
    print(f'RMSE for baseline model: {rmse_train_mu:.08}')
    # r^2 for the baseline will always be 0 by definition
    print('R^2 for baseline model: 0.0')

# -------------------------------------------------------------

def get_model_polynomial(X_train, X_validate, X_test,
                         y_train, y_validate, y_test,
                         f_features):
    '''
    This function will create new dataframes with features transformed for use with a
    polynomial regression. It will then create a linear regression ml model, fit on the
    training data, and make preditions using the model.
    '''
    # Create the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2) #quadratic function

    # Fit and transform X_train
    X_train_degree2 = pf.fit_transform(X_train[f_features])
    # Transform X_validate & X_test 
    X_validate_degree2 = pf.transform(X_validate[f_features])
    X_test_degree2 = pf.transform(X_test[f_features])

    # make a linear regression model
    lm2 = LinearRegression()
    # fit the model on the training data
    lm2.fit(X_train_degree2, y_train.quality)
    # usage the model on the training data
    y_train['quality_pred_lm2'] = lm2.predict(X_train_degree2)
    # Evaluate: RMSE
    rmse_train = mean_squared_error(y_train.quality, 
                                    y_train.quality_pred_lm2, squared=False)

    # repeat usage on validate
    y_validate['quality_pred_lm2'] = lm2.predict(X_validate_degree2)
    # evaluate: RMSE
    rmse_validate = mean_squared_error(y_validate.quality, 
                                       y_validate.quality_pred_lm2, squared=False)
    # caluculate r_2 value
    r_2 = explained_variance_score(y_validate.quality,
                                   y_validate.quality_pred_lm2)
    # print the model metrics
    print('model : Polynomial-kbest')
    print(f'RMSE_train: {rmse_train:.4}')
    print(f'RMSE_validate: {rmse_validate:.4}')
    print(f'difference: {rmse_validate - rmse_train:.4}')
    print(f'R2: {r_2:.4}')

    # return the model and modified polynomial features for use on test data
    return lm2, X_train_degree2, X_validate_degree2, X_test_degree2

# -------------------------------------------------------------
    
# if we want to use test on the polynomial model use this
def get_polynomial_test(lm2, X_test_degree2, y_test):
    '''
    This function will take in a linear ml model and transformed polynomial features
    data and will use the model on the provided test data.
    '''
    # make predictions on the test data
    y_test['quality_pred_lm2'] = lm2.predict(X_test_degree2)
    # Evaluate: RMSE
    rmse_test = mean_squared_error(y_test.quality, 
                                   y_test.quality_pred_lm2, squared=False)
    # calculate r^2 value
    r_2 = explained_variance_score(y_test.quality,
                                       y_test.quality_pred_lm2)
    # print metrics
    print('Polynomial Model on Test Data')
    print(f'RMSE on test data: {rmse_test:.08}')
    print(f'R^2 value: {r_2:0.4}')
    # return test df modified with predictions
    return y_test

# -------------------------------------------------------------

def get_pred_error_plot(y_test):
    '''
    This function will take in a DataFrame containing the actual quality scores 
    and predicted quality scores generated from the test dataset, it will then
    display a plot the error of the wine quality predictions
    '''
    # set figure size
    plt.figure(figsize=(16,12))
    # create a line at zero error
    plt.axhline(label="No Error")
    # create a scatter plot with the error amounts
    plt.scatter(y_test.quality, (y_test.quality_pred_lm2_kbest - y_test.quality), 
                alpha=.5, color="grey", s=100, label="Model 2nd degree Polynomial")
    # change the x and y labels and label sizes
    plt.xlabel('Actual Wine Quality', size=14)
    plt.ylabel('Error of Predicted Wine Qualities', size=14)
    # add a title to the plot
    plt.title('Prediction Error of Polynomial Regression Model', size=16)
    # create a legend
    plt.legend(loc=1)
    # display the plot
    plt.show()