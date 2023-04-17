# import data acquisition functions, visualization tools, and 
import wrangle as w
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd

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
    # reset index
    train = train.reset_index(drop=True)
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
#_______________________________



def get_corr_heatmap(train):
    corr_matrix = train.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Heat Map of Correlation')
    plt.show()


#_______________________________
	
def find_k(X_train, cluster_vars, k_range):
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)

        # X[0] is our X_train dataframe..the first dataframe in the list of dataframes stored in X. 
        kmeans.fit(X_train)
#i think this part might need to be X_train[cluster_vars],
# otherwise it is looking at clustering all of the columns together

        
        
        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_) 

    # compute the difference from one k to the next
    delta = [round(sse[i] - sse[i+1],0) for i in range(len(sse)-1)]

    # compute the percent difference from one k to the next
    pct_delta = [round(((sse[i] - sse[i+1])/sse[i])*100, 1) for i in range(len(sse)-1)]

    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    k_comparisons_df = pd.DataFrame(dict(k=k_range[0:-1], 
                             sse=sse[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # plot k with inertia
    plt.plot(k_comparisons_df.k, k_comparisons_df.sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('The Elbow Method to find the optimal k\nFor which k values do we see large decreases in SSE?')
    plt.show()

    # plot k with pct_delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.pct_delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Percent Change')
    plt.title('For which k values are we seeing increased changes (%) in SSE?')
    plt.show()

    # plot k with delta
    plt.plot(k_comparisons_df.k, k_comparisons_df.delta, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Absolute Change in SSE')
    plt.title('For which k values are we seeing increased changes (absolute) in SSE?')
    plt.show()

    return k_comparisons_df


#_______________________________
	
def cluster_columns(scaled_df, n_clusters):
    # Initialize an empty dictionary to store the cluster labels for each column
    clusters_dict = {}

    # Loop over each column in the scaled DataFrame
    for col in scaled_df.columns:
        # Initialize a k-means model with n_clusters
        kmeans = KMeans(n_clusters=n_clusters)

        # Fit the model to the column data
        kmeans.fit(scaled_df[col].values.reshape(-1, 1))

        # Get the cluster labels for each data point in the column
        col_clusters = kmeans.labels_

        # Add the cluster labels to the clusters_dict with the column name as the key
        clusters_dict[col] = col_clusters

    # Create a new DataFrame from the clusters_dict
    clusters_df = pd.DataFrame(clusters_dict)

    # Return the clusters_df
    return clusters_df

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