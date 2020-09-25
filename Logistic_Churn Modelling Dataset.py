import pandas as pd
import numpy as np
churn=pd.read_csv("D:\\LIVE WIRE - DS\\Data Set\\Churn_Modelling.csv")
#checking null values
churn.isna().sum()
x=churn.iloc[:,:13]#independent
y=churn.iloc[:,13]#dependent

#converting strings into int
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
churn.columns
#salary.dropna().sum()#removing null values.

#dealing with string type of col.
churn_cols=['Surname','Geography', 'Gender']
for i in churn_cols:
    churn[i]=l.fit_transform(churn[i])   

#dealing with int.type of value
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
churn_cols=['RowNumber', 'CustomerId','Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']
for i in churn_cols:
    churn[i]=mm.fit_transform(churn[i].values.reshape(-1,1))

# Target
X=churn[['RowNumber', 'CustomerId', 'Surname', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']]

y=churn[['CreditScore']]

 #train test spit
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.2, random_state=0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train,y_train)

from sklearn.metrics import accuracy_score
accuracy_score(model,y_test)


model.predict(X_test)
model.predict_proba(X_test)
pred=model.predict(y_test)
model.score(X_test,y_test)
#0.2