# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S.Prema Latha
RegisterNumber: 212222230112

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
data.head():

![Screenshot 2023-10-23 092233](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/a491bf65-f59b-4b30-9ef3-570457dbd185)

data.info():

![Screenshot 2023-10-23 092238](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/94090c41-eba5-449f-a4e0-42ea36533e24)

isnull() & sum() function:

![Screenshot 2023-10-23 092243](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/6739982b-997a-4fcd-b33c-2fa83c90c379)

data.head() for position:

![Screenshot 2023-10-23 092250](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/44664ef2-56c7-4201-af75-0536c98733e1)

MSE value:

![Screenshot 2023-10-23 092256](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/1a816b72-f1b2-45bc-9b0b-de51422c47ca)

R2 value:

![Screenshot 2023-10-23 092300](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/0d7b4632-db4e-4de7-a71b-cd51d78a23a2)

Prediction value:

![Screenshot 2023-10-23 092308](https://github.com/premalatha-sureshbabu/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120620842/8b5eb781-d0b7-483d-bcd5-d973e1e5d9d4)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
