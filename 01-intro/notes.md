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

