#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Data Collection

# In[6]:


# loading the dataset to a Pandas DataFrame
df = pd.read_csv('Wine_Dataset.csv')


# In[7]:


# number of rows & columns in the dataset
df.shape


# In[8]:


# first 5 rows of the dataset
df.head()


# In[9]:


# checking for missing values
df.isnull().sum()


# Data Analysis and Visulaization

# In[10]:


# statistical measures of the dataset
df.describe()


# In[11]:


# number of values for each quality
sns.catplot(x='quality', data = df, kind = 'count')


# In[12]:


# volatile acidity vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'volatile acidity', data = df)


# In[13]:


# citric acid vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y = 'citric acid', data = df)


# Correlation

# 1. Positive Correlation
# 2. Negative Correlation

# In[14]:


correlation = df.corr()


# In[15]:


# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':8}, cmap = 'Blues')


# Data Preprocessing

# In[16]:


# separate the data and Label
X = df.drop('quality',axis=1)


# In[17]:


X.head()


# Label Binarizaton

# In[18]:


Y = df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)


# In[19]:


Y.head()


# Train & Test Split

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[21]:


print(Y.shape, Y_train.shape, Y_test.shape)


# Model Training:
# 
# Random Forest Classifier

# In[22]:


model = RandomForestClassifier()


# In[23]:


model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[24]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[25]:


print('Accuracy : ', test_data_accuracy)


# Building a Predictive System

# In[26]:


input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')

