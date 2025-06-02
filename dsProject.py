import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


data= pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Python programming(sem4)\\python files\\Titanic-Dataset.csv")
print(data.head())
data=data.drop(['Name','Cabin','Ticket'],axis=1)
print(data.head())
print(data.info())
print(data.isnull().sum())

age=data['Age'].mean()

data['Age']=data['Age'].fillna(age)
data['Embarked']= data['Embarked'].fillna(data['Embarked'].mode()[0])   # first most frequent value
print(data.isnull().sum())
data=pd.get_dummies(data, columns=['Sex','Embarked'],drop_first=True)   #Logistic Regression only works with numbers.
                                          #drop_first-> A boolean (True or False) to drop one of the dummy columns per original feature.
x=data.drop('Survived',axis=1)
y=data['Survived']

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=45)
model=LogisticRegression(max_iter=200) #The max number of iterations 
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


print("Classification Report: \n",classification_report(y_test,y_pred))
print("\n")
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("\n")
print("Accuracy Score: \n",accuracy_score(y_test,y_pred))   #your model got 82.7% of the test samples right.