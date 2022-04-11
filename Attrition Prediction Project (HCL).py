#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# In[2]:


data = pd.read_csv('C:\\Users\\srira\\Downloads\\HR-Analytics-and-Employee-Attrition-Prediction-master\\HR-Analytics-and-Employee-Attrition-Prediction-master\\Datasets\\WA_Fn-UseC_-HR-Employee-Attrition.csv')
data = data.drop(columns=['StandardHours','EmployeeCount','Over18','EmployeeNumber','StockOptionLevel'])

le = preprocessing.LabelEncoder()
categorial_variables = ['Attrition','BusinessTravel','Department','EducationField',
                        'Gender','JobRole','MaritalStatus','OverTime']
for i in categorial_variables:
    data[i] = le.fit_transform(data[i])
data.head(5)
data.to_csv('LabelEncoded_CleanData.csv')


# In[3]:


data.to_csv


# In[16]:


train.info()


# In[5]:


target = data['Attrition']
train = data.drop('Attrition',axis = 1)
train.shape


# In[6]:


train.size


# In[7]:


train.describe()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


def plot_corr(data,size=10):
    
    corr=data.corr()
    fig,ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr.columns)), corr.columns)
    

plot_corr(data)


# In[136]:


plt.title("PercentSalaryHike vs Satisfication level")

sns.barplot(x='PercentSalaryHike',y='JobSatisfaction',data=train)


# In[52]:


plt.figure(figsize=(15,8))
plt.title("MonthlyIncome vs Satisfication level")
sns.scatterplot(x=train['JobSatisfaction'],y=train['MonthlyIncome'],hue='JobLevel',data=data, palette='cool')
plt.show()


# In[12]:


train_accuracy = []
test_accuracy = []
models = ['Logistic Regression','SVM','KNN','Decision Tree','K Means Clustering']


# In[13]:


#Defining a function which will give us train and test accuracy for each classifier.
def train_test_error(y_train,y_test):
    train_error = ((y_train==Y_train).sum())/len(y_train)*100
    test_error = ((y_test==Y_test).sum())/len(Y_test)*100
    train_accuracy.append(train_error)
    test_accuracy.append(test_error)
    print('{}'.format(train_error) + " is the train accuracy")
    print('{}'.format(test_error) + " is the test accuracy")


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size=0.33, random_state=42)


# In[15]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)
train_predict = log_reg.predict(X_train)
test_predict = log_reg.predict(X_test)
y_prob = log_reg.predict(train)
y_pred = np.where(y_prob > 0.5, 1, 0)
train_test_error(train_predict , test_predict)


# In[17]:


from sklearn import svm
SVM = svm.SVC(probability=True)
SVM.fit(X_train,Y_train)
train_predict = SVM.predict(X_train)
test_predict = SVM.predict(X_test)
train_test_error(train_predict , test_predict)


# In[18]:


from sklearn import neighbors
n_neighbors = 15
knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(X_train,Y_train)
train_predict = knn.predict(X_train)
test_predict = knn.predict(X_test)
train_test_error(train_predict , test_predict)


# In[19]:


from sklearn import tree
dec = tree.DecisionTreeClassifier()
dec.fit(X_train,Y_train)
train_predict = dec.predict(X_train)
test_predict = dec.predict(X_test)
train_test_error(train_predict , test_predict)


# In[20]:


from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2, random_state=1)
kms.fit(X_train,Y_train)
train_predict = kms.predict(X_train)
test_predict = kms.predict(X_test)
train_test_error(train_predict,test_predict)


# In[21]:


results = DataFrame({"Test Accuracy" : test_accuracy , "Train Accuracy" : train_accuracy} , index = models)


# In[22]:


results


# In[103]:


model_scores={'Logistic Regression':log_reg.score(X_test,Y_test),
             'KNN classifier':knn.score(X_test,Y_test),
             'Support Vector Machine':SVM.score(X_test,Y_test),
              'Decision tree':dec.score(X_test,Y_test)
             }
model_scores


# In[104]:


model_compare=pd.DataFrame(model_scores,index=['accuracy'])
model_compare


# In[135]:


model_compare.T.plot(kind='line')
sns.set(rc={'axes.facecolor':'pink'})


# In[ ]:




