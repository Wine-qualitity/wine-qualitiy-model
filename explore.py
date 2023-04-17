# import data acquisition functions, visualization tools, and 
import wrangle as w
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



#_______________________________

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
        
#_______________________________

def get_plot_alcohol_by_quantity(train):
    '''
    This function will show a plot of alcohol content by wine quality.
    '''
    # set figure size
    plt.figure(figsize=(16,12))
    # create the plot
    sns.lmplot(data=train, x='alcohol', y='quality', 
               # add a line showing the correlation
               line_kws={'color': 'red'})
    # add a title
    plt.title('As Alcohol Content Increases, Quality Also Increases', size=15)
    # add axis labels
    plt.xlabel('Alcohol Content in the Wine', size=14)
    plt.ylabel('Quality Score of the Wine', size=14)
    plt.annotate('correlation line', (13.2,6.5))
    # add a legend
    # show the plot
    plt.show()
    
#_______________________________

def get_plot_density_by_quantity(train):
    '''
    This function will show a plot of density by wine quality.
    '''
    # set figure size
    plt.figure(figsize=(16,12))
    # create the plot
    sns.lmplot(data=train.sample(1000), x='density', y='quality', palette='blues',
               # add a line showing the correlation
               line_kws={'color': 'red'})
    # add a title
    plt.title('As Density Increases, Quality Decreases', size=15)
    # add axis labels
    plt.xlabel('Density of the Wine', size=14)
    plt.ylabel('Quality Score of the Wine', size=14)
    # add a label for the correlation line
    plt.annotate('correlation line', (.999,5.5))
    # show the plot
    plt.show()

#_______________________________
    
def get_plot_chlorides_by_quantity(train):
    '''
    This function will show a plot of chlorides by wine quality.
    '''
    # set figure size
    plt.figure(figsize=(16,12))
    # create the plot
    sns.lmplot(data=train.sample(1000), x='chlorides', y='quality', palette='blues',
               # add a line showing the correlation
               line_kws={'color': 'red'})
    # add a title
    plt.title('As Chlorides Increases, Quality Decreases', size=15)
    # add axis labels
    plt.xlabel('Chlorides of the Wine', size=14)
    plt.ylabel('Quality Score of the Wine', size=14)
    # add a label for the correlation line
    plt.annotate('correlation line', (.15,5.5))
    # show the plot
    plt.show()
    
#_______________________________

def get_corr_heatmap(train):
    corr_matrix = train.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Heat Map of Correlation')
    plt.show()


#_______________________________
	



def get_x_y_train_val_test(train,validate,test):
    '''
    This function will return 
    X_train, y_train, X_validate, y_validate, X_test and y_test
    
    it will drop quality and types of wine if you want the type of wine it must
    be defined else where 
    
    '''

    # make a list of columns i want to drop 
    x_drop_cols = ['quality', 'type_of_wine_white']

    # drop and split train data
    X_train = train.drop(columns= x_drop_cols)
    y_train= train.quality 

    # drop and split validate data

    X_validate = validate.drop(columns= x_drop_cols)
    y_validate = validate.quality


    # Drop and split test data 

    X_test = test.drop(columns= x_drop_cols)
    y_test = test.quality 

    return X_train, y_train, X_validate, y_validate, X_test, y_test




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
	

	
	
	
	
def apply_kmeans(data,k):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(data)

    data[f'k_means_{k}'] = kmeans.labels_
    
    return data

  # _______________________________

def best_kmeans(data,k_max):
    
    '''
    EXAMPLE USEAGE
    
    data = scaled_train[['alcohol', 'quality']]
    
    best_kmeans(data,k_max=10)
    
     will produce elbow graph with clusters
    '''
   


    means = []
    
    inertia = []
    
    for k in range(1, k_max):
        kmeans = KMeans(n_clusters=k)
        
        kmeans.fit(data)
        
        means.append(k)
        
        inertia.append(kmeans.inertia_)
        
        fig =plt.subplots(figsize=(10,5))
        plt.plot(means,inertia, 'o-')
        plt.xlabel('means')
        plt.ylabel('inertia')
        plt.grid(True)
        plt.show()



	 # _______________________________
