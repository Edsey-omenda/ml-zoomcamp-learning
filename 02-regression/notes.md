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
#### Pandas attributes and methods:

- df[col].unique() -> return a list of unique values in the series
- df[col].nunique() -> return the number of unique values in the series
- df.isnull().sum() -> return the number of null values in the dataframe
#### Matplotlib and seaborn methods:

- %matplotlib inline -> assure that plots are displayed in jupyter notebook's cells
- sns.histplot() -> show the histogram of a series
#### Numpy methods:

- np.log1p() -> apply log transformation to a variable, after adding one to each input value.
Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible.

### 2.4 Setting up the validation framework.
In general, the dataset is splitted into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and vectors of targets (y). First, the size of the partitions is calculated. Next, the records are shuffled to ensure that the values in the three partitions contain non-sequential records from the dataset. Finally, the partitions are created using the shuffled indices.

#### Pandas attributes and methods:

- df.iloc[] -> return subsets of records of a dataframe, being selected by numerical indices
- df.reset_index() -> restate the orginal indices
- del df[col] -> eliminate a column variable
#### Numpy methods:

- np.arange() -> return an array of numbers
- np.random.shuffle() -> return a shuffled array
- np.random.seed() -> set a seed for reproducibility

### 2.5 Linear regression.
