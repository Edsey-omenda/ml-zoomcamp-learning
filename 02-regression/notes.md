# 2. Machine Learning for Regression

- 2.1 Car price prediction project
- 2.2 Data preparation
- 2.3 Exploratory data analysis
- 2.4 Setting up the validation framework
- 2.5 Linear regression
- 2.6 Linear regression: vector form
- 2.7 Training linear regression: Normal equation
- 2.8 Baseline model for car price prediction project
- 2.9 Root mean squared error
- 2.10 Using RMSE on validation data
- 2.11 Feature engineering
- 2.12 Categorical variables
- 2.13 Regularization
- 2.14 Tuning the model
- 2.15 Using the model
- 2.16 Car price prediction project summary
- 2.17 Explore more

### 2.1 Car price prediction project

This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from this kaggle competition.

### Project plan:

- Prepare data and Exploratory data analysis (EDA)
- Use linear regression for predicting price
- Understanding the internals of linear regression
- Evaluating the model with RMSE
- Feature engineering
- Regularization
- Using the model

### 2.2 Data preparation.

Pandas attributes and methods:

- pd.read_csv(<file_path_string>) -> read csv files
- df.head() -> take a look of the dataframe
- df.columns -> retrieve colum names of a dataframe
- df.columns.str.lower() -> lowercase all the letters
- df.columns.str.replace(' ', '\_') -> replace the space separator
- df.dtypes -> retrieve data types of all features
- df.index -> retrieve indices of a dataframe

### 2.3 Exploratory data analysis.
