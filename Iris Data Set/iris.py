import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Iris.csv")
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
print(df.shape)
sns.pairplot(df)
plt.show()
sns.scatterplot(x=df["SepalLengthCm"],y=df["SepalWidthCm"],hue=df["Species"])
plt.show()
X=df.drop("Species",axis=1)
y=df["Species"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25,random_state=15)
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(y_train)
y_test=label_encoder.transform(y_test)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
linear=LinearRegression()
linear.fit(X_train_scaled,y_train)
y_pred=linear.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("mean_absolute_error :", mae)
print("mean_squared_error :",mse)
print("r2_score :",r2_score)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train_scaled,y_train)
y_pred_svr = svr.predict(X_test_scaled)

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("mean_absolute_error :", mae)
print("mean_squared_error :",mse)
print("r2_score :",r2_score)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.svm import SVC
svc=SVC()
svr.fit(X_train_scaled,y_train)
y_pred=linear.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("mean_absolute_error :", mae)
print("mean_squared_error :",mse)
print("r2_score :",r2_score)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid={
    "C":[0.1,10,100,1000],
    "gamma":[1,0.1,0.002],
    "kernel":["rbf","linear"]
}
grid=GridSearchCV(estimator=SVR(),param_grid=param_grid,n_jobs=-1,verbose=3)
grid.fit(X_train_scaled,y_train)
grid.best_params_
y_pred=svr.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("mean_absolute_error :", mae)
print("mean_squared_error :",mse)
print("r2_score :",r2_score)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

svc = SVC()
svc.fit(X_train_scaled, y_train)

y_pred = svc.predict(X_test_scaled)

print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))
print("classification_report:\n", classification_report(y_test, y_pred))
print("accuracy_score:\n", accuracy_score(y_test, y_pred))

