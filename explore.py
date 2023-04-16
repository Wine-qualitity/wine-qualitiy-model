# import data acquisition functions
import wrangle as w
# import data tools
import numpy as np
import pandas as pd
# import visualization tools
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def check_hypothesis(data, x, y, α=0.05, test=stats.pearsonr):
    '''
    This function will check the provided x and y variables from the 
    provided dataset (data) for statistical relevence according 
    to a pearsonsr test (this is changable by entering the desired test as a kwarg)
    '''
    # run the requested statistical test on variables x and y from data
    r, p = test(data[x], data[y])
    # if the resulting p-value is less than alpha, then reject the null hypothesis
    if p < α:
        # print results rejecting null hypothesis
        print(f"Since the p-value is less than {α}, \n\
we can reject the null hypothesis and conclude that {x} and {y} are correlated.")
        print(f"The correlation coefficient between \
{x} and {y} is {r:.2f} with a p-value of {p:.4f}")
        print('_______________________________________________________')
    # if p-value >= alpha, then we fail to reject the null hypothesis
    else:
        # print the results failing to reject the null hypothesis
        print(f"Since the p-value is greater than or equal to {α}, \n\
we fail to reject the null hypothesis and conclude \n\
that there is insufficient evidence to suggest a correlation between {x} and {y}.")
        print('_______________________________________________________')
        
        
def get_plot_alcohol_by_quantity(train):
    '''
    This function will show a plot of alcohol content by wine quality.
    '''
    # set figure size
    plt.figure(figsize=(16,12))
    # create the plot
    sns.relplot(data=train.sample(1000), x='alcohol', y='quality')
    # add a regression line
    plt.axhline(color='red') #not right
    # add a title
    plt.title('As Alcohol Content Increases, Quality Also Increases', size=15)
    # add axis labels
    plt.xlabel('Alcohol Content in the Wine', size=14)
    plt.ylabel('Quality Score of the Wine', size=14)
    # show the plot
    plt.show()