# import
import Acquire as a 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Functions to use for our data 





def prepare(df):
    '''
    This function cleans the dataframe and replaces spaces with underscores
    '''
    df = pd.get_dummies(df, columns=['type_of_wine'], drop_first=True)
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    # returns the dataframe with the cleaned columns and one-hot encoded columns
    return df



def split(df):
    '''
    This function splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test
                                       


def overview(df):
    '''
    print shape of DataFrame, .info() method call, and basic descriptive statistics via .describe()
    parameters: single pandas dataframe, df
    return: none
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))
    
    
def hypothesis_test(data, x, y, alpha=0.05):
    # Perform Pearson correlation test
    r, p = stats.pearsonr(data[x], data[y])
    print(f"The Pearson correlation coefficient between {x} and {y} is {r:.2f} with a p-value of {p:.2f}")

    # Determine whether to accept or reject null hypothesis
    if p < alpha:
        print(f"Since the p-value is less than {alpha}, we can reject the null hypothesis and conclude that {x} and {y}           are correlated.")
        print('_______________________________________________________')
    else:
        print(f"Since the p-value is greater than or equal to {alpha}, we fail to reject the null hypothesis and conclude         that there is insufficient evidence to suggest a correlation between {x} and {y}.")
        print('_______________________________________________________')




