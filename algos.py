# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mydataset = pd.read_csv('CSM_2014_2015_dataset.csv')
mydataset.info()
mydataset.isna().sum()
new_df = mydataset.interpolate()



Ratings = new_df["Ratings"].values
Review = []
for num in Ratings:
    if num < 5:
        Review.append("Bad")
    elif num >=7:
        Review.append("Good")
    else:
       Review.append("Average")
     
new_df['Review'] = Review   



Gross = new_df['Gross'].values
Budget = new_df['Budget'].values
commercial_success = [0]*len(Gross)
for i in range(0,len(Gross)) :
    num = (Gross[i] - Budget[i]) 
    if num < 10000000:
        commercial_success[i] = "flop"
    elif num > 17000000:
        commercial_success[i] ="Super hit"
    else:
        commercial_success[i] = "hit"
new_df['commercial_success'] =  commercial_success



Likes = new_df['Likes'].values
Dislikes = new_df['Dislikes'].values
critical_success = [0]*len(Gross)
for i in range(0,len(Likes)) :
    num = Likes[i] / Dislikes[i]
    if num < 15:
        critical_success[i] = "not supported by public"
    else:
        critical_success[i] = "supported by public"
new_df['critical_success'] =  critical_success



print('Prediction for review')
y = new_df['Review']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y) 
 ## good=2, average=0, bad=1
 
x = new_df[['Gross', 'Views' , 'Likes' , 'Aggregate Followers']]
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)

##logical regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
predicted = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test , predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))



##naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
predicted=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test,predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))




## SVm
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0,gamma='auto')
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
p=classifier.predict(x_test)
cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,p),4) *100))
print("Classification Report\n"+str(classification_report(y_test, p)))




## KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
acc=[]
for i in range(1,21):
    classifier1=KNeighborsClassifier(n_neighbors=i)
    classifier1.fit(x_train,y_train)
    predicted1=classifier1.predict(x_test)
    acc.append(accuracy_score(y_test, predicted1)*100)
plt.figure(figsize=(7,7))
plt.plot(acc)
plt.xticks(np.array(range(1,17)))
plt.title("Selection of k")
plt.xlabel("k for Nearest Neighbour")
plt.ylabel("Accuracy")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(accuracy_score(y_test, p)*100))
print("Classification Report\n"+str(classification_report(y_test, p)))





    
print('Prediction for commercial_success')    
###


##logical regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
predicted = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test , predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))



##naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
predicted=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test,predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))




## SVm
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0,gamma='auto')
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
p=classifier.predict(x_test)

cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,p),4) *100))
print("Classification Report\n"+str(classification_report(y_test, p)))




## KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
acc=[]
for i in range(1,21):
    classifier1=KNeighborsClassifier(n_neighbors=i)
    classifier1.fit(x_train,y_train)
    predicted1=classifier1.predict(x_test)
    acc.append(accuracy_score(y_test, predicted1)*100)
plt.figure(figsize=(7,7))
plt.plot(acc)
plt.xticks(np.array(range(1,17)))
plt.title("Selection of k")
plt.xlabel("k for Nearest Neighbour")
plt.ylabel("Accuracy")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(accuracy_score(y_test, p)*100))
print("Classification Report\n"+str(classification_report(y_test, p)))




print('Prediction for critical_success')
####
y = new_df['critical_success']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y) 
 ##  not=0, yes=1 
y = new_df['critical_success']
x = new_df[['Gross', 'Views' , 'Likes' , 'Aggregate Followers']]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 0)


##logical regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
predicted = log_reg.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test , predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))



##naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
predicted=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

cm=confusion_matrix(y_test,predicted)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,predicted),4) *100))
print("Classification Report\n"+str(classification_report(y_test, predicted)))




## SVm
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0,gamma='auto')
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
p=classifier.predict(x_test)
cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(round(accuracy_score(y_test,p),4) *100))
print("Classification Report\n"+str(classification_report(y_test, p)))




## KNN
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train,y_train)
p=classifier.predict(x_test)
acc=[]
for i in range(1,21):
    classifier1=KNeighborsClassifier(n_neighbors=i)
    classifier1.fit(x_train,y_train)
    predicted1=classifier1.predict(x_test)
    acc.append(accuracy_score(y_test, predicted1)*100)
plt.figure(figsize=(7,7))
plt.plot(acc)
plt.xticks(np.array(range(1,17)))
plt.title("Selection of k")
plt.xlabel("k for Nearest Neighbour")
plt.ylabel("Accuracy")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm=confusion_matrix(y_test,p)
print(cm)
print("Accuracy is "+str(accuracy_score(y_test, p)*100))
print("Classification Report\n"+str(classification_report(y_test, p)))
