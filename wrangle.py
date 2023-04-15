# Our standard imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# This is the main function that acquires the data
def acquire_data():
    '''
   This function acquires the data from the data world and combines it into a single dataframe.
'''
    # First the function checks to see if the file exists
    
    if os.path.exists('wine_data.csv'):
        df = pd.read_csv('wine_data.csv')
    # If the file doesn't exist, then the function acquires the data from the data world
    else:
        df_1 = pd.read_csv('https://query.data.world/s/s7hsvetdhmtlkdq4x6ofz3wd5snwut?dws=00000')
        df_2 = pd.read_csv('https://query.data.world/s/dth4xlpnfu3lnyln2lxfdms4vxf3hl?dws=00000')
        # we add the type of wine to the dataframe. 
        
        df_1['type_of_wine'] = ['Red'] * len(df_1)
        df_2['type_of_wine'] = ['White'] * len(df_2)
        # create a list of the two dataframes
        combine = [df_1,df_2]
        # Then we concatenate the two dataframes into the final dataframe
        df = pd.concat(combine)
        df.to_csv('wine_data.csv',index=False)
    # the function returns the dataframe
    return df

def prepare(df):
    '''
    This function cleans the dataframe and replaces spaces with underscores
    '''
    # Here we set the df.columns to be replaces with the columsn with underscores instead of spaces to help data miniplualtion in pandas.
    df.columns = df.columns.str.replace(' ', '_')
    
    # returns the dataframe with the cleaned columns
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
    
#     Here we are spliting train into .8 of the original dataset. and test into 20% of the original dataset.
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    
    # here we assign validate to be .3 of the train dataset 
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    # returns train validate and test dataframes
    return train, validate, test
                                       
def wrangle():
    '''
    This function will perform acquisition, cleaning and spliting of the dataset via one command
    '''
    df = split(
         prepare(
             acquire_data()))
    return df

def overview(df):
    '''
    print shape of DataFrame, .info() method call, and basic descriptive statistics via .describe()
    parameters: single pandas dataframe, df
    return: none
    '''
    #prints the shape of the dataframe
    print('--- Shape: {}'.format(df.shape))
    # prints the info method call
    print('--- Info')
    df.info()
    # # prints the describe() method call
    print('--- Column Descriptions')
    print(df.describe(include='all'))
    
    





