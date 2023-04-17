# Our standard imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler





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
        
        df_1['type_of_wine'] = ['red'] * len(df_1)
        df_2['type_of_wine'] = ['white'] * len(df_2)
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
    train, validate, test = split(prepare(acquire_data()))
    
    return train, validate, test






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
    
    
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['fixed_acidity', 'volatile_acidity', 'citric_acid',
                                 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                                 'total_sulfur_dioxide', 'density', 'ph',
                                 'sulphates', 'alcohol'],
               scaler=MinMaxScaler(),
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(train[columns_to_scale]),
        columns=train[columns_to_scale].columns.values, 
        index = train.index)
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(validate[columns_to_scale]),
        columns=validate[columns_to_scale].columns.values).set_index(
        [validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(
        test[columns_to_scale]), 
        columns=test[columns_to_scale].columns.values).set_index(
        [test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
	
	
	

	
def get_null_percentage(df):

	return round(((df.isna().sum()) / len(df)) * 100, 2)