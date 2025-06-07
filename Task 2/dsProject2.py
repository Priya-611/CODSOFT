# Random Forest Regressor

# "Random" because it uses random parts of data and random features. "Forest" because it combines many Decision Trees
# used to predict numbers
# A Decision Tree splits your data into smaller and smaller parts to make a prediction.
# best for Complex relationships (non-linear)


#Regression → Predicting numbers
#Split criterion: Mean Squared Error (MSE)
#Evaluation metrics: R² Score, Mean Absolute Error (MAE), MSE, RMSE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  #Random Forest can learn complex patterns because it builds rule-based decision trees that capture feature combinations automatically.
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Loading dataset
df=pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\Python programming(sem4)\\python files\\IMDb Movies India.csv", encoding='ISO-8859-1')
print(df.head())
print(df.isnull().sum())
print(df.info())


df=df.drop(['Name'],axis=1)   #dropping the whole duration column
df=df.dropna(subset=['Votes','Rating'])   #dropping rows where these values are null

#Filling missing value
#converting into numeric
#replacing string part  
df['Votes'] = df['Votes'].str.replace(",", "").astype(int)
df['Duration']=df['Duration'].str.replace(" min","")
df['Duration']=df['Duration'].apply(pd.to_numeric)
df['Duration']=df['Duration'].fillna(df['Duration'].mean())


df['Year']=df['Year'].str.replace("-","")
df['Year']=df['Year'].str.replace(")","")
df['Year']=df['Year'].str.replace("(","")
df['Year']=df['Year'].apply(pd.to_numeric)

print(df.info())

#filling missing values by "Other"
df['Genre']=df['Genre'].fillna("Other")
df['Director']=df['Director'].fillna("Other")
df['Actor 1']=df['Actor 1'].fillna("Other")
df['Actor 2']=df['Actor 2'].fillna("Other")
df['Actor 3']=df['Actor 3'].fillna("Other")
print(df.isnull().sum())


#Model training
x=df.drop(['Rating'],axis=1)
x=pd.get_dummies(x,drop_first=True)
y=df['Rating']

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=45)
model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mse=mean_squared_error(y_test,y_pred) # average difference between predicted and actual ratings (lower is better)
r_sq=r2_score(y_test,y_pred) # ranges from 0 to 1 (Closer to 1 is better[perfect prediction])

#Evaluation
print("Mean_sqaures_error: ",mse)  #1.1777727935606062
print("r2_Square: ",r_sq)    #0.3608033059464323


#Mean Squared Error (MSE): ~1.19  This indicates that on average, the square of the error between the predicted rating and the actual rating is 1.19 units².
# R² Score: ~0.35 This means the model explains 35% of the variance in the IMDb ratings on the test set.


