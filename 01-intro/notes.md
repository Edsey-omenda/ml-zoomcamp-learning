# Introduction to Machine Learning

### Topics

###### 1. What is Machine Learning.

###### 2. ML vs Rule-Based Systems.

###### 3. Supervised Machine Learning.

###### 4. CRISP-DM.

###### 5. Model Selection Process.

###### 6. Setting up the Environment.

###### 7. Introduction to NumPy.

###### 8. Linear Algebra Refresher.

###### 9. Introduction to Pandas.

### 1.1 What is Machine Learning.

Machine Learning is the process of extracting patterns from data which is of two types: 1. Features: Information about the object. 2. Target: Property to predict for unseen objects.

A model is the output of a machine learning.
Training a model means taking the features and the target and giving them to a machine learning algorithm.The algorithm produces a model.

### 1.2 Machine Learning vs Rule-Based Systems

Traditional Rule-Based systems are based on a set of characteristics(keywords, email length, etc.) that identify an email as either spam or not spam. As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintainance as the system grows.
ML solves this problem with the steps below:

    1. Get Data - Emails from the user's spam folder and inbox give examples of spam and not spam.
    2. Define and Calculate feaures - Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).
    Each email can be encoded (converted) to the values of its features and target.
    3. Train and use the model - A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The predictions are probabilities, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.

### 1.3 Supervised Machine Learning

Supervised machine learning is about teaching an algorithm by showing it examples.
A Model is always trained and then it can make predictions on new features. In this way, the model is taught by certain features and targets: 1. Feature matrix (X): All characteristics of the object we want to make predictions for (made of observations or objects(rows) and features(columns)) 2. Target variable (y): The vector y is the target we want to learn (a vector with target information we want to predict. For each row of X there's a value in y)

The model can be represented as a function, **g**, that takes the feature matrix, **X**, as input and tries to predict values as close as possible to the targets, **y**. The process of finding this function **g** is called **training**.
So g(X) is approximated to be equal to the output y.

#### Types of SML problems.

1. Regression: The output is a number, could be from zero to infinity e.g a car's price.
2. Classification: The outputis a category.
   A.) Binary: There are two categories.
   B.) Multiclass problems: There are more than two categories.
3. Ranking: The output is the top scores associated with correponding items. It is applied in recommender systems.

### 1.4 CRISP-DM

CRISP-DM stands for Cross-Industry Standard Process for Data Mining. It describes an iterative process for carrying out data mining and machine learning projects.

It involves six steps:

1. **Business understanding:** Identify the business problem and whether it needs ML. The project's goal should be measurable.
2. **Data understanding:** Analyse available data sources and decide whether more data is required.
3. **Data preparation:** Clean and transform the data into a format suitable for the model, using reproducible pipelines.
4. **Modeling:** Train and compare different models. Results may reveal a need for new features or fixes to data issues.
5. **Evaluation:** Assess model performance and whether it meets the business goal.
6. **Deployment:** Put the solution into use, monitor its performance, and maintain it. Deployment can include online evaluation.

Consider how easy the project will be to maintain.

ML projects require many iterations.

**Iteration**

- Start simple.
- Learn from the feedback.
- Improve.

### 1.5 Model Selection Process

Which models to use?

- Logistic Regression.
- Decision Tree.
- Random Forest.
- Neural Network.

The validation dataset is not used in training. There are feature matrices and y vectors for both training and validation datasets. The model is fitted with training data, and it is used to predict the y values of the validation feature matrix. Then, the predicted y values (probabilities) are compared with the actual y values.

**Multiple Comparisons Problem (MCP)**: Just by chance one model can be lucky and obtain good predictions because all of them are probabilistic.
The test set can help to avoid the MCP. Obtaining the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best.

1. Split datasets in training, validation, and test. E.g. 60%, 20% and 20% respectively.
2. Train the models.
3. Evaluate the models.
4. Select the best model.
5. Apply the best model to the test dataset.
6. Compare the performance metrics of validation and test.

**NB**: Note that it is possible to reuse the validation data. After selecting the best model (step 4), the validation and training datasets can be combined to form a single training dataset for the chosen model before testing it on the test set.

### 6. Setting up the Environment.
Completed ✅

### 7. Introduction to NumPy.
Numpy, short for Numerical Python, is a powerful Python library that enables efficient and convenient array manipulation and mathematical operations. It forms the foundation for many scientific and data-related tasks. 

#### Importing Numpy.
import numpy as np

#### Creating Arrays.
Arrays are the building blocks of Numpy, and they can be thought of as lists but with enhanced features.

#### Creating Arrays with Zeros, Ones, or Constants.
You can create arrays filled with zeros, ones, or any constant using np.zeros(), np.ones(), and np.full():
- zeros_array = np.zeros(10)
- ones_array = np.ones(10)
- constant_array = np.full(10, 3)

#### Converting Lists to Arrays.
To convert a Python list into a Numpy array, you can use np.array():
- my_list = [2, 3, 4]
- array_from_list = np.array(my_list)

#### Generating Ranges of Numbers.
Numpy provides functions for generating arrays of sequential numbers. For example:
range_array = np.arange(10)  # Creates an array from 0 to 9

#### Creating Arrays with Linear Spacing.
np.linspace() creates arrays with evenly spaced numbers within a specified range:
linspace_array = np.linspace(0, 1, 11)  # Creates 11 numbers from 0 to 1

#### Multi-dimensional Arrays.
Numpy can handle multi-dimensional arrays, often referred to as matrices. Here are some examples:
- zeros_matrix = np.zeros((5, 2))
- ones_matrix = np.ones((5, 2))
- constant_matrix = np.full((5, 2), 3)

#### Indexing and Slicing Arrays.
Like Python lists, you can access elements in Numpy arrays using indexing and slicing. For two-dimensional arrays:
- arr = np.array([[2, 3, 4], [4, 5, 6]])
- first_row = arr[0]      # Gets the first row
- first_col = arr[:, 0]  # Gets the first column

For random numbers from a normal distribution or integers within a range:
- normal_distribution = np.random.randn(5, 2)
- random_integers = np.random.randint(low=0, high=100, size=(5, 2))

### Array Operations.
Numpy excels in performing mathematical operations on arrays efficiently.

#### Element-wise Operations.
You can perform operations on entire arrays element by element:
- arr = arr + 1   # Adds 1 to each element
- arr = arr * 2   # Multiplies each element by 2
        # Similar operations for division and exponentiation

#### Element-wise Operations with Two Arrays.
You can also perform operations between two arrays of the same shape:
- arr1 = np.ones(4)
- arr2 = np.full(4, 3)
- result = arr1 + arr2  # Element-wise addition
- result = arr1 / arr2  # Element-wise division

#### Comparison Operations.
You can perform element-wise comparisons and create boolean arrays:
- arr = np.array([1, 2, 3, 4])
- greater_than_2 = arr > 2  # Produces [False, False, True, True]

#### Selecting Elements Based on Conditions.
You can create subarrays based on certain conditions:
- selected_elements = arr[arr > 1]  # Gets elements greater than 1

#### Summary Operations.
Numpy provides functions for summarizing array data:
- min_value = arr.min()    # Minimum value
- max_value = arr.max()    # Maximum value
- sum_value = arr.sum()    # Sum of all elements
- mean_value = arr.mean()  # Mean (average) value
- std_deviation = arr.std()  # Standard deviation

In conclusion, Numpy is an essential library for anyone working with numerical data in Python. It simplifies array creation, manipulation, and mathematical operations, making it a powerful tool for scientific computing and data analysis.