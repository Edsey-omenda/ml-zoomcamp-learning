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
