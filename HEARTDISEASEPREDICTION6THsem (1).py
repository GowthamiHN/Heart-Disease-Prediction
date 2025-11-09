#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sn


# In[2]:


#Load the csv data to a pandas DataFrame
heart_data=pd.read_csv('heartdiseasedata.csv')


# In[3]:


#print first 5 rows of the dataset
heart_data.head()


# In[4]:


#print last 5 rows of the dataset
heart_data.tail()


# In[5]:


heart_data.shape


# In[6]:


#getting some info about the data
heart_data.info()


# In[7]:


#checking for missing values
heart_data.isnull().sum()


# In[8]:


#statistical measures about the data
heart_data.describe()


# In[9]:


x=heart_data.drop(columns='target',axis=1)
y=heart_data['target']


# In[10]:


print(x)


# In[11]:


print(y)


# In[12]:


#spliting a dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[13]:


print(x.shape,x_train.shape,x_test.shape)


# In[14]:


#data visualization
import matplotlib.pyplot as plt
heart_data=pd.read_csv('heartdiseasedata.csv')
heart_data.hist(edgecolor='blue',linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()


# In[15]:


plt.figure(figsize=(20,12))
sn.set_context('notebook',font_scale=1.3)
sn.heatmap(heart_data.corr(),annot=True,linewidth=2)
plt.tight_layout()


# In[16]:


plt.figure(figsize=(25,12))
sn.set_context('notebook',font_scale=1.5)
sn.barplot(x=heart_data.age.value_counts()[:10].index,y=heart_data.age.value_counts()[:10].values)
plt.tight_layout()


# In[17]:


#logistic Regression
model=LogisticRegression()


# In[18]:


#training the LogisticRegression model with Training data
model.fit(x_train,y_train)


# In[19]:


#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[20]:


print('accuracy on training data:',training_data_accuracy)


# In[21]:


#accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('accuracy on test data:',test_data_accuracy)


# In[22]:


input_data=(54,0,1,132,288,1,0,159,1,0,2,1,2)

#change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')

