# Title - Wine Quality Predictions


# Project Description - Describes what your project is and why it is important

Wine is one of the most popular drinks worldwide. There are many wines produced and this project can help determine some of the factors that determine if a wine is considered good or bad. Wine quality also has a large impact on how much can be charged for the wine by a restaurant or winery.


# Project Goal Clearly states what your project sets out to do and how the information gained can be applied to the real world

This project will be focused on predicting a wine quality score based on various given factors such as acidity, sugar amount, and alcohol content. We will be attempting to identify which variables have the most effect on a wine's quality score. We will also be building a machine learning model that will attempt to predict a wine's quality score based on our identified variables. We will approaching this project as a linear regression problem, since our target variable is numerical value.


# Initial Hypotheses - Initial questions used to focus your project
What are some


# Project Plan - Guides the reader through the different stages of the pipeline as they relate to your project

- Planning - The steps required to be taken during this project will be laid out in this readme file. There will also be a Trello Kanban chart describing and tracking each step of the project.

- Acquisition - Data will be acquired from https://data.world/food/wine-quality which is a publicly available dataset. Once the files have been downloaded, a local version will be created and stored locally by our acquire.py script.

- Preparation - We will do inital exploration of our data to determine if there are outliers or null values. If the dataset contains outliers or nulls, we will make determinations on what to do with each based on the effect on the overall dataset. We will rename columns in order to make them easier to understand or work with. If there are any data types that are not best suited for the data, we will change the data types. We will also be splitting our data into train, validate and test groups to limit potential data poisoning. Since we are approaching the project as a regression problem, we will also be scaling our data.

- Exploration - We will explore the unscaled data to find statistically valid correlations to our target variable. We will be creating at least 4 visualizations to help us determine correlations and variables that could be used for clustering. We will also look for variables that could be better understood by converting the variable into bins.

- Modeling - We will be approaching the problem as a linear regression problem. Therefore we will be making multiple models using regression algorithms such as Ordinary Least Squares (OLS), LASSO + LARS, Generalized Linear Model (Tweedie), and Polynomial Regression. We will be creating a baseline model using the mean of known quality scores. We will be evaluating our models using the Root Mean Squared Error (RMSE) and goodness of fit (R^2) metrics.

Delivery - We will be packaging our findings in a final_report.ipynb file. We will also be creating a Canvas slide presentation using our findings and presenting them as if we were presenting to a winery supply chain marketing department on behalf of the California Wine Institute.


# Data Dictionary - Gives a definition for each of the features used in your report and the units they are measured in, if applicable


# Steps to Reproduce - Gives instructions for reproducing your work. i.e. Running your notebook on someone else's computer.

You will need an account for data.world in order to acquire the dataset. Once the data has been acquired, you will need to have all .py files contained in the same local directory that has the final_report.ipynb file.
