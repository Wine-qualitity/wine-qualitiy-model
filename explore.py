# import data acquisition functions, visualization tools, and 
import wrangle as w
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


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

def get_plot_chlorides_by_quality(train):
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
    
def get_plot_density_by_quality(train):
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

def get_plot_alcohol_by_quality(train):
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
    
def get_plot_volatile_acidity_by_quality(train):
    # create a hexbin plot of volatile acidity by quality
    plt.hexbin(data=train.sample(500), x='volatile_acidity', 
               y='quality', gridsize=8, cmap='PuBu')
    # add a regression line
    reg = linregress(train.volatile_acidity, train.quality)
    plt.axline(xy1=(0, reg.intercept), slope=reg.slope, color="r")
    # annotate the regression line
    plt.annotate('regression line', (0.8, 4.5))
    # add a title and resize it
    plt.title('As Volatile Acidity goes up, Quality goes down', size=16)
    # add axis labels and resize
    plt.ylabel('Quality of the Wine', size=14)
    plt.xlabel('Volatile Acidity of the Wine', size=14)
    # display the plot
    plt.show()
    
#_______________________________

def get_corr_heatmap(train):
    '''
    This function will display a heatmap of the potential correlations between variables in 
    our dataset
    '''
    # get the correlation values
    corr_matrix = train.corr()
    # create a plot
    plt.figure(figsize=(10,10))
    # plot a heatmap of the correlations
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # add a title
    plt.title('Heat Map of Correlation')
    # display the plot
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
    # return the new X and y datasets
    return X_train, y_train, X_validate, y_validate, X_test, y_test

#_______________________________

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
    
    # fit the scaler object
    scaler.fit(train[columns_to_scale])
    # applying the scaler to the training data:
    train_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(train[columns_to_scale]),
        columns=train[columns_to_scale].columns.values, 
        index = train.index)
    # use the scaler object on the validation data                                              
    validate_scaled[columns_to_scale] = pd.DataFrame(
        scaler.transform(validate[columns_to_scale]),
        columns=validate[columns_to_scale].columns.values).set_index(
        [validate.index.values])
    # use the scaler on the test data
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(
        test[columns_to_scale]), 
        columns=test[columns_to_scale].columns.values).set_index(
        [test.index.values])
    # if the scaler object was requested in the function call,
    # then return the scaler object with the scaled datasets
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    # otherwise return the scaled datasets
    else:
        return train_scaled, validate_scaled, test_scaled
    
# _______________________________

def best_kmeans(data,k_max):
    '''
    EXAMPLE USEAGE:
    data = scaled_train[['alcohol', 'quality']]
    best_kmeans(data,k_max=10)
    
    This function will produce an elbow graph with clusters
    '''
    # create empty list variables to store results
    means = []
    inertia = []
    # cycle through our desired amount of k's
    for k in range(1, k_max):
        # create a KMeans object with current k
        kmeans = KMeans(n_clusters=k)
        # fit the kmeans object to our data
        kmeans.fit(data)
        # store the kmeans object in our means list
        means.append(k)
        # store the inertia for current k in the inertia list
        inertia.append(kmeans.inertia_)
        # create a figure
        fig =plt.subplots(figsize=(10,5))
        # plot the current k and inertia
        plt.plot(means,inertia, 'o-')
        # add axis labels
        plt.xlabel('means')
        plt.ylabel('inertia')
        # remove gridlines
        plt.grid(True)
        # display the plot
        plt.show()
        
# _______________________________

def apply_kmeans(data,k):
    '''
    This function will create a clusters based on the given variables and data
    '''
    # create a kmeans object with k clusters
    kmeans = KMeans(n_clusters=k)
    # fit the kmeans object on our data
    kmeans.fit(data)
    # store the clustered data as a new column
    data[f'k_means_{k}'] = kmeans.labels_
    # return the modified dataset
    return data

# _______________________________

def plot_kmeans_histogram():
    '''
    This will create a histogram showing alcohol content grouped by number of wines in each cluster
    '''
    # gather the data used
    train, validate, test= w.split(w.prepare(w.acquire_data()))
    train_scaled, validate_scaled, test_scaled = w.scale_data(train, 
    validate, 
    test)
    # create new variables for the alcohol content clusters
    alcohol_lvl = train_scaled[['alcohol']]
    alcohol_lvl = apply_kmeans(data= alcohol_lvl,k=3)
    alcohol_lvl_dict = {
        0: 'medium',
        1: 'low',
        2: 'high'
    }
    # add the cluster names to the dataset
    alcohol_lvl['clusters'] = alcohol_lvl['k_means_3'].map(alcohol_lvl_dict)
    # create a plot showing the clusters
    fig = sns.histplot(data=alcohol_lvl, x=alcohol_lvl['alcohol'], 
                       hue=alcohol_lvl.clusters, palette="YlOrBr")
    # add a title
    plt.title('K Means Distribution of Alcohol', size=25)
    # add axis labels
    plt.xlabel('Alcohol content', size=25)
    plt.ylabel('Wine Count', size=25)
    # change the legend labels
    plt.legend(labels=['high', 'low', 'medium'])
    # save the plot (commented out since we don't need to save the plot every time it runs)
#     plt.savefig('histogram_clusters_on_alcohol.png')
    # display the plot
    plt.show()

# _______________________________

def give_hypothesis_alcohol():
    '''
    This function will run a 2-sample statistical test to check if wines with high alcohol content
    have a higher quality score than wines with lower alcohol content
    '''
    # gather the required data
    train, validate, test= w.split(w.prepare(w.acquire_data()))
    train_scaled, validate_scaled, test_scaled = w.scale_data(train, 
               validate, 
               test)
    # create clusters
    alcohol_lvl = train_scaled[['alcohol']]
    alcohol_lvl = apply_kmeans(data= alcohol_lvl,k=3)
    alcohol_lvl_dict = {
        0: 'medium',
        1: 'low',
        2: 'high'
    }
    # add the clusters to the datasets
    alcohol_lvl['clusters'] = alcohol_lvl['k_means_3'].map(alcohol_lvl_dict)
    # create variables for each cluster
    low_alc = alcohol_lvl[alcohol_lvl['clusters'] == 'low']
    med_alc = alcohol_lvl[alcohol_lvl['clusters'] == 'medium']
    high_alc = alcohol_lvl[alcohol_lvl['clusters'] == 'high']
    # combine the low and med clusters in preparation for stats test
    combine_me = [low_alc,med_alc,]
    not_high_alc = pd.concat(combine_me, axis=0)
    not_high_alc['quality'] = train['quality']
    high_alc['quality'] = train['quality']

    # isoslate quality from both df
    qual_not_high_alc = not_high_alc.quality
    qual_high_alc = high_alc.quality

    # set our alpha 
    alpha = 0.05
    # run a 2-sample t-test
    t, p = stats.ttest_ind(qual_high_alc, qual_not_high_alc, equal_var=False)
    t, p / 2
    # check if p / 2 is greater than the alpha, or if t is negative
    # if so then we FAIL to reject the null hypothesis
    if p / 2 > alpha:
        print("We fail to reject null hypothesis")
    elif t < 0:
        print("We fail to reject null hypothesis")
    # otherwise we REJECT the null hypothesis
    else:
        print("We reject null hypothesis")