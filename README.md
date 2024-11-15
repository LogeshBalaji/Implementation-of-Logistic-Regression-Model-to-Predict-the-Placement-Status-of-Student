# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOGESH B
RegisterNumber: 24900577 
*/
```
import pandas as pd  
data=pd.read_csv(r"Placement_Data.csv")  
data.head()  
data1=data.copy()  
data1=data1.drop(["sl_no","salary"],axis=1)  
data1.head()  
data1.isnull().sum()  
data1.duplicated().sum()  
from sklearn.preprocessing import LabelEncoder  
le=LabelEncoder()  
data1["gender"]=le.fit_transform(data1["gender"])  
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])  
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])  
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])  
data1["degree_t"]=le.fit_transform(data1["degree_t"])  
data1["workex"]=le.fit_transform(data1["workex"])  
data1["specialisation"]=le.fit_transform(data1["specialisation"])  
data1["status"]=le.fit_transform(data1["status"])  
data1  
x=data1.iloc[:,:-1]  
x  
y=data1["status"]  
y  
from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)  
from sklearn.linear_model import LogisticRegression  
lr=LogisticRegression(solver="liblinear")  
lr.fit(x_train,y_train)  
y_pred=lr.predict(x_test)  
y_pred  
from sklearn.metrics import accuracy_score  
accuracy=accuracy_score(y_test,y_pred)  
accuracy  
from sklearn.metrics import confusion_matrix  
confusion=confusion_matrix(y_test,y_pred)  
confusion  
from sklearn.metrics import classification_report  
classification_report1=classification_report(y_test,y_pred)  
print(classification_report1)  
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])  

## Output:
# HEAD
![HEAD](https://github.com/user-attachments/assets/a9452526-c279-4bf0-8773-82216a9aee0a) 
# COPY
![COPY](https://github.com/user-attachments/assets/803939e2-3489-4dd9-9723-d537ea46541a)  
![Screenshot 2024-11-10 212604](https://github.com/user-attachments/assets/f2241f84-ec0f-401e-8578-b9fed5cefd60)  
# FIT TRANSFORM
![FIT TRANSFORM](https://github.com/user-attachments/assets/f01b85cb-e183-4596-b6de-565f0a20e9e4)  
# LOGISTIC REGRESSION
![Screenshot 2024-11-10 212314](https://github.com/user-attachments/assets/3391dcf6-39b1-4497-8aa2-f789319c5ba6)
![Screenshot 2024-11-10 212258](https://github.com/user-attachments/assets/bf00eb14-2eae-4f51-8c5d-ac45bca36472)  
# ACCURACY SCORE
![Screenshot 2024-11-10 212240](https://github.com/user-attachments/assets/c48f372d-2343-4b95-bc3b-d5912ef2490d)  
# CONFUSION MATRIX
![Screenshot 2024-11-10 212219](https://github.com/user-attachments/assets/b5a19887-64a6-4993-bf3c-9d8560c6bc56)   
# CLASSFICATION REPORT
![Screenshot 2024-11-10 212202](https://github.com/user-attachments/assets/de06118f-3726-408a-802c-bdf19d22363e)   
# PREDICTION
![Screenshot 2024-11-10 212144](https://github.com/user-attachments/assets/20419204-a461-43f2-af2b-bc8fc9bba659)   
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
