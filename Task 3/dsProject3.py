# Random Forest Classifier
# A machine learning algorithm used for classification tasks.It builds many decision trees and uses majority voting to decide the final class
# Predicting categories (e.g., Setosa, Versicolor)
# Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  #Random Forest can learn complex patterns because it builds rule-based decision trees that capture feature combinations automatically.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

#Loading dataset
df=pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Python programming(sem4)\\python files\\IRIS.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

#replacing "Iris-" as it contibuting nothing
df['species']=df['species'].str.replace("Iris-","")

#Model Training
x=df.drop(['species'],axis=1)
y=df['species']

x_train, x_test,y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=45)
model=RandomForestClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


#Evaluation
print("\n")
print("CLassification Report: \n",classification_report(y_test,y_pred))
print("\n")
print("Confusion_matrix: \n",confusion_matrix(y_test,y_pred))
print("\n")
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))  #model got 93% of accuracy


# Classication Report output:
# Precision->How many predicted were actually correct (low false positives)
# Recall ->	How many actual were correctly predicted (low false negatives)
# F1-score ->	Harmonic mean of precision & recall (best when balanced)
# Macro avg	-> Simple average (all classes treated equally)
# Weighted avg ->Takes into account the support (number of samples in each class)

# macro_precision = (precision_setosa + precision_versicolor + precision_virginica) / 3
# weighted_precision = (precision_setosa * 11 + precision_versicolor * 7 + precision_virginica * 12) / 30

