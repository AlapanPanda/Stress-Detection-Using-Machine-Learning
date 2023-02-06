#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[4]:


df=pd.read_csv("Z:/Stress Detection Using ML/Week 3/spam.csv",encoding="latin-1")


# In[6]:


#visualizing dataset
df.head(n=10)


# In[7]:


df.shape


# In[8]:


# to check target attribute is binary or not
np.unique(df['class'])


# In[9]:


np.unique(df['message'])


# In[10]:


# creating spars matrix
x=df["message"].values
y=df["class"].values

#create count vectorizer object
cv=CountVectorizer()

x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[11]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[17]:


#spliting train +test 3:1

train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[20]:


bnb=BernoulliNB(binarize=0.0)
#0.0- is counterizing vector
model=bnb.fit(train_x,train_y)

y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[22]:


#priting training score
print(bnb.score(train_x,train_y)*100)

#printing testing score
print(bnb.score(test_x,test_y)*100)


# In[23]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[24]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[2]:


user=input("Enter the text") 
import cv
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)

