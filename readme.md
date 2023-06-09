# Wine Quality Predictions
- By Adam Harris and Fermin Garcia


# Project Description

Wine is one of the most popular drinks worldwide. There are many wines produced and this project can help determine some of the factors that determine if a wine is considered good or bad. Wine quality also has a large impact on how much can be charged for the wine by a restaurant or winery.


# Project Goal

This project will be focused on predicting a wine quality score based on various given factors such as acidity, sugar amount, and alcohol content. We will be attempting to identify which variables have the greatest effect on a wine's quality score. We will also be building a machine learning model that will attempt to predict a wine's quality score based on our identified variables. We will be approaching this project as a linear regression problem, since our target variable is numerical value.


# Initial Hypotheses

- Does alcohol content affect the quality score of the wine?
- Does the density of the wine affect the quality?
- Does the number of chlorides in the wine affect the quality?
- If we cluster the alcohol content into groups, will wines with high alcohol content have a higher quality score on average than wines with lower alcohol content?
- Does volatile acidity level have a correlation with quality of the wine?

- H$_0$: There is no correlation between the quality of wine and individual features.
- H$_a$:There is a correlation between quality of wine and individual features. 


# Project Plan

- Planning - The steps required to be taken during this project will be laid out in this readme file. There will also be a Trello Kanban chart describing and tracking each step of the project.

- Acquisition - Data will be acquired from https://data.world/food/wine-quality which is a publicly available dataset. Once the files have been downloaded, the data will be combined into one local csv file by our acquire.py script.

- Preparation - We will do inital exploration of our data to determine if there are outliers or null values. If the dataset contains outliers or nulls, we will make determinations on what to do with each based on the effect on the overall dataset. This is a rare instance where we decided to keep our duplicates. We made this determination based off visual review of our duplicates, while there are some across the board, some of the numerical values that extend to four decimal places are identical. This lead us to conclude that some scores are destined to be duplicated as wine must meet specific range of qualifications to qualify as wine, so duplicate rows are sure to be duplicated. We will rename columns in order to make them easier to understand or work with. If there are any data types that are not best suited for the data, we will change the data types. We will also be splitting our data into train, validate and test groups to limit potential data poisoning. Since we are approaching the project as a regression problem, we will also be scaling our data.

- Exploration - We will explore the unscaled data to find statistically valid correlations to our target variable. We will be creating at least 4 visualizations to help us determine correlations. We will also be looking at combinations of variables that could be useful for clustering. We will also look for variables that could be better understood by converting the variable into bins.

- Modeling - We will be approaching the problem as a linear regression problem. Therefore we will be making multiple models using regression algorithms such as Ordinary Least Squares (OLS), LASSO + LARS, Generalized Linear Model (Tweedie), and Polynomial Regression. We will be creating a baseline model using the mean of known quality scores from our training dataset. We will be evaluating our models using the Root Mean Squared Error (RMSE) and goodness of fit (R^2) metrics.

- Delivery - We will be packaging our findings in a final_report.ipynb file. We will also be creating a Canvas slide presentation using our findings and presenting them as if we were presenting to a winery supply chain marketing department on behalf of the California Wine Institute.


# Data Dictionary

## Wine data dictionary.

| Field                 | Description                                                       |
|-----------------------|-------------------------------------------------------------------|
| Fixed Acidity         | Fixed acidity of the wine (g/l)                                    |
| Volatile Acidity      | Volatile acidity of the wine (g/l)                                 |
| Citric Acid           | Citric acid of the wine (g/l)                                      |
| Residual Sugar        | Residual sugar of the wine (g/l)                                   |
| Chlorides             | Chlorides of the wine (g/l)                                        |
| Free Sulfur Dioxide   | Free sulfur dioxide of the wine (mg/l)                             |
| Total Sulfur Dioxide  | Total sulfur dioxide of the wine (mg/l)                            |
| Density               | Density of the wine (g/ml)                                         |
| pH                    | pH value of the wine                                               |
| Sulphates             | Sulphates of the wine (g/l)                                        |
| Alcohol               | Alcohol content of the wine (% vol.)                               |
| Quality               | Quality rating of the wine on a scale of 0 to 10 (median of at least 3 evaluations made by wine experts) |
| Type_of_wine          | The type of wine red or white                                      |


# Steps to Reproduce

You will need to create an account for data.world in order to acquire the dataset. Once the data has been acquired, you will need to have the wrangle.py, explore.py and modeling.py files contained in the same local directory that has the final_report.ipynb file. Run the final_report file.


# Data Citation 

## The citation for the wine quality dataset is:

	Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016