# CODSOFT
This repository contains data science internship tasks including Titanic survival prediction using Logistic Regression


## Titanic Survival Prediction Using Logistic Regression

This project is part of the **CODSOFT Data Science Internship** and focuses on building a machine learning model to predict whether a passenger on the Titanic would have survived or not. The solution leverages the Logistic Regression algorithm and uses a structured dataset containing information such as age, gender, ticket class, fare, and more.


### Project Objective

The main objective of this project is to apply data preprocessing techniques, perform exploratory data analysis (EDA), and develop a predictive model using logistic regression. This classic binary classification problem helps in understanding fundamental concepts in machine learning, such as feature engineering, model evaluation, and metrics analysis.


### Dataset Description

The dataset used in this project is the famous Titanic dataset, which is available on [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset). It contains data about the passengers aboard the Titanic ship, with features including:

- **PassengerId**: Unique ID for each passenger
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: Survival status (0 = No, 1 = Yes - this is the target variable)


### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

These tools were used for data manipulation, visualization, and building the machine learning model.


### Exploratory Data Analysis

The following steps were performed to clean and explore the data:

- Removal of irrelevant columns such as `Name`, `Cabin`, and `Ticket`
- Handling missing values in `Age` by filling with the mean
- Filling missing values in `Embarked` with the most frequent value (mode)
- Converting categorical variables (`Sex`, `Embarked`) into numerical format using one-hot encoding
- Splitting the dataset into training and testing sets
- Visualizing data distributions and survival rates using Seaborn and Matplotlib


### Model Development

We used the **Logistic Regression** model from the Scikit-learn library to perform binary classification. Key steps include:

- Training the model on 80% of the data
- Testing on 20% of the data
- Evaluating performance using:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-Score)


### Results
The model achieved an accuracy of **~82%**, which indicates that it can reliably predict survival outcomes for Titanic passengers.

