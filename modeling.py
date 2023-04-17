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


def get_baseline_model(y_train):
    '''
    This function will print the mean baseline RMSE and R^2 scores
    '''
    y_train['quality_pred_mean'] = y_train.quality.mean()
    rmse_train_mu = mean_squared_error(y_train.quality,
                                   y_train.quality_pred_mean, squared=False)
    print('Baseline Model (mean)')
    print(f'RMSE for baseline model: {rmse_train_mu:.08}')
    print('R^2 for baseline model: 0.0')


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
    
    # change the x and y tick labels and size to be more readable
#     plt.xticks(ticks=[0,200_000,400_000,600_000,800_000,1_000_000], 
#                labels=['0', '200,000', '400,000', '600,000', '800,000', '1,000,000'],
#                size = 12)
#     plt.yticks(size=12,
#                ticks=[600_000, 400_000, 200_000, 0, -200_000, -400_000, 
#                       -600_000, -800_000, -1_000_000],
#                labels=['600,000', '400,000', '200,000', '0', '-200,000', '-400,000', 
#                       '-600,000', '-800,000', '-1,000,000'])
    # change the x and y labels and label sizes
    plt.xlabel('Actual Wine Quality', size=14)
    plt.ylabel('Error of Predicted Wine Qualities', size=14)
    # add a title to the plot
    plt.title('Prediction Error of Polynomial Regression Model', size=16)
    # create a legend
    plt.legend(loc=1)
    # display the plot
    plt.show()