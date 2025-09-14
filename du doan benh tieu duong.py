import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df = pd.read_csv(r"C:\Users\ACER\Desktop\kagglr\diabetes.csv")

print (df.head())
print (df.info())

print (df['Outcome'].value_counts())
sns.countplot(x='outcome',data=df)
plt.show()

x=df.drop("Outcome",axis=1)
y=df["Outcome"]

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)


lr=LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)

dt=DecisionTreeClassifier(random_state=42)
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)

rf=RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)

models={
    "logistic Regression":y_pred_lr,
    "Decision Tree":y_pred_dt,
    "Random Forest":y_pred_rf
}
for name,pred in models.items():
    print (f"/n{name}")
    print ("Accuracy:",accuracy_score(y_test,pred))
    print ("classification report:/n",classification_report(y_test,pred))

new_data=np.array([[2,120,70,20,79,25,0,0.351,29]])
new_data_scaled=scaler.transform(new_data)
prediction=rf.predict(new_data_scaled)
print ("Ket qua du doan:","co benh " if prediction[0]==1 else "khong benh")    
