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

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is the feature matrix X and a y vector of predictions is obtained, trying to be as close as possible to the actual y values. The linear regression formula is the sum of the bias term ( w0), which refers to the predictions if there is no information, and each of the feature values times their corresponding weights as ( xi1⋅w1 + xi2⋅w2 + ...+ xin⋅wn).

So the simple linear regression formula looks like:
g(xi) = w0 + xi1⋅w1 + xi2⋅w2 + ... + xin⋅wn
And that can be further simplified as:
g(xi) = w0 + ∑ sum of wi.xij where j=1 and goeas all the way to n.
w0 = 7.1
def linear_regression(xi):

    n = len(xi)

    pred = w0
    w = [0.01, 0.04, 0.002]
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred

If we look at the equation ∑ sum of wi.xij where j=1 and goeas all the way to n.we know that this is nothing else but a vector-vector multiplication. Hence, we can rewrite the equation as
g(xi) = w0 + x.T/i.w
We need to assure that the result is shown on the untransformed scale by using the inverse function exp().

### 2.6 Linear regression: vector form.

So this is linear regression. To summarize how we got here: we started with a for loop for one car, recognized that it is a dot product between the vector of features and the vector of weights, made it even shorter by adding the fictional feature - the virtual 1 - to the feature vector, and finally generalized it to a complete feature matrix, where linear regression is nothing else but a matrix-vector multiplication.

You may have been wondering where the weights come from - how do we set the values for them? That is what we will talk about in the next lesson.

So this is linear regression. To summarize how we got here: we started with a for loop for one car, recognized that it is a dot product between the vector of features and the vector of weights, made it even shorter by adding the fictional feature - the virtual 1 - to the feature vector, and finally generalized it to a complete feature matrix, where linear regression is nothing else but a matrix-vector multiplication.

You may have been wondering where the weights come from - how do we set the values for them? That is what we will talk about in the next lesson.

When all the records are included, the linear regression can be calculated with the dot product between feature matrix and vector of weights, obtaining the y vector of predictions.

### 2.7 Training linear regression: Normal equation.

Obtaining predictions as close as possible to
y
target values requires the calculation of weights from the general LR equation. The feature matrix does not have an inverse because it is not square, so it is required to obtain an approximate solution, which can be obtained using the Gram matrix (multiplication of feature matrix (X) and its transpose (X^T)). The vector of weights or coefficients w obtained with this formula is the closest possible solution to the LR system.

Normal Equation:

w = (X^T.X)^-1.X^T.y
where:
X^T.X is the Gram Matrix

### 2.8 Baseline model for car price prediction project.

- In this lesson we build a baseline model and apply the df_train dataset to derive weights for the bias (w0) and the features (w). For this, we use the train_linear_regression(X, y) function from the previous lesson.
- Linear regression only applies to numerical features. Therefore, only the numerical features from df_train are used for the feature matrix.
- We notice some of the features in df_train are nan. We set them to 0 for the sake of simplicity, so the model is solvable, but it will be appropriate if a non-zeo value is used as the filler (e.g. mean value of the feature).
- Once the weights are calculated, then we apply them on g(X) =w0 + X⋅w to derive the predicted y vector.
- Then we plot both predicted y and the actual y on the same histogram for a visual comparison.

### 2.9 Root Mean Squared Error (RMSE).

- In the previous lesson we found out our predictions were a bit off from the actual target values in the training dataset. We need a way to quantify how good or bad the model is. This is where RMSE can be of help.
- Root Mean Squared Error (RMSE) is a way to evaluate regression models. It measures the error associated with the model being evaluated. This numerical figure can then be used to compare models, enabling us to choose the one that gives the best predictions.

RMSE = sqrt(1/m . ∑ (g(xi) - yi)\*\*2) where i starts at 1 and goes all the way to m.

- g(xi) is the prediction
- yi is the actual value
- m is the number of observations in the dataset (i.e. cars)

### 2.10 Computing RMSE on validation data.

Calculation of the RMSE on validation partition of the dataset of car price prediction. In this way, we have a metric to evaluate the model's performance.
A lot of things are happening here, so let's write a special function for that. We'll call it prepare_X, and it takes a dataframe:

def prepare_X(df):
df_num = df[base]
df_num = df_num.fillna(0)
X = df_num.values
return X

The function does three things. First it selects the numerical columns, then it fills the missing values, and finally it extracts the feature matrix - the NumPy array - and returns it. We went from one line to five, but at least it is easier to understand what is going on.

The point of this function is that we prepare the data in the same way regardless of whether it is the training, validation or test dataset. It works on any dataframe - that is why the argument is just df and not df_train.

Now we can train the model and validate it. First the training part: we prepare the matrix from the training dataframe and train the model. Then the validation part: we prepare the validation matrix with the same function, apply the model, and compute the RMSE between the actual target values and our predictions:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

The number is pretty similar to what we had on the training data. That is what we want to see: the model behaves on unseen data about as well as on the data it learned from.

Looking at the code, we can see the two parts clearly:

- The training part only touches the training dataset: we prepare the matrix and learn the weights.
- The validation part prepares the validation dataset in the same way as the training dataset, then applies the model learned in the previous step, and computes the root mean squared error for that model.

### 2.11 Simple feature engineering.

Feature engineering is the process of creating new features from the ones we already have. In this unit we create our first new feature - the age of a car - and see it improve the RMSE of our model quite a bit.

Let's modify the prepare_X function we wrote in the previous unit. The new feature is called age, and we compute it in the same way we just did, except that instead of df_train we use the dataframe the function receives - it could be the training, validation or test set.

We also need to use the new feature when building the matrix. Instead of base, we create a list called features that contains the baseline numerical features plus the new age feature:

def prepare_X(df):
df['age'] = 2017 - df.year
features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

When we run this function, the first line adds a new column - and if we look at df_train again, we see that a new column appeared in it. We modified the dataframe we passed in.

This is not something this function should do. As a user of this function, I don't want it to change my data - what if it does something that cannot be undone? It is much better if the function doesn't modify the dataframes it receives.

The fix is simple: before doing anything, we take a copy of the dataframe and work with that copy inside the function. The original dataframe stays unchanged

- after the function returns, we simply forget about the copy, because all we care about at the end is X.

def prepare_X(df):
df = df.copy()

    df['age'] = 2017 - df.year
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

Now X_train has six columns - the five base features plus age, which is the last one.

#### Checking the improvement.

Let's validate the model with the same code as before: prepare the matrices, train the model, apply it to the validation set and compute the RMSE:

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

The model improved, and it is quite an improvement - the RMSE went down from 0.76 to 0.51:

We can see that it is a big improvement by doing the same thing as previously: plotting the predicted values and the actual values on the same histogram. Note that now we compare against y_val, because we are predicting on the validation dataset, not on the training dataset as previously.

sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
sns.histplot(y_val, label='target', color='blue', alpha=0.5, bins=50)
plt.legend()

The shapes of the two distributions are now much closer. There is still a lot of room for improvement - for example, the model completely misses this bar here - but in general it is doing much better.

### 2.12 Categorical variables.
Categorical variables are typically represented as strings, and pandas identifies them as object types. However, some variables that appear to be numerical may actually be categorical (e.g., the number of doors a car has). All these categorical variables need to be converted to a numerical form because ML models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necessarily all of them. This transformation from categorical to numerical variables is known as One-Hot encoding.

### 2.13 Regularization.
If the feature matrix has duplicate columns (or columns that can be expressed as a linear combination of other columns), it will not have an inverse matrix. But, sometimes this error could be passed if certain values are slightly different between duplicated columns.

So, if we apply the normal equation with this feature matrix, the values associated with duplicated columns are very large, which decreases the model performance. To solve this issue, one alternative is adding a small number to the diagonal of the feature matrix, which corresponds to regularization.

This technique works because the addition of small values to the diagonal makes it less likely to have duplicated columns. The regularization value is a hyperparameter of the model. After applying regularization the model performance improved.

#### Linear combination
I mentioned the term linear combination in the video, but didn't explain what it means. So if you're interested what it means, you can read here

One column is a linear combination of others when you can express one column of a matrix as a sum of others columns
The simplest example is when a column is an exact duplicate of another column
Another example. Let's say we have 3 columns: a, b, c. If c = 0.2 * a + 0.5 * b, then c is a linear combination of a and b

#### Ridge Regression
The regularization technique used (adding a factor to the diagonals of Gram Matrix) in this lesson is Ridge Regression.