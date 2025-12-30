#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# In[3]:


df = pd.read_csv('covid_toy.csv')


# In[4]:


df.head()


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(df.drop(columns = ['has_covid']), df['has_covid'], test_size = 0.2)


# In[7]:


x_train


# In[8]:


df.isnull().sum()


# In[9]:


x_train


# # 1.Aam zindagi

# In[10]:


# adding a fever column

si = SimpleImputer()
x_train_fever = si.fit_transform(x_train[['fever']])

# also the test data as welll

x_test_fever = si.fit_transform(x_test[['fever']])

x_train_fever.shape


# In[34]:


oe = OrdinalEncoder(categories = [['Mild','Strong']])
x_train_cough = oe.fit_transform(x_train[['cough']])

x_test_cough = oe.fit_transform(x_test[['cough']])

x_train_cough.shape


# In[29]:


# one hot encoding on gender and city

hot = OneHotEncoder(drop = 'first', sparse_output = False)
x_train_gender_city = hot.fit_transform(x_train[['gender','city']])

x_test_gender_city = hot.fit_transform(x_test[['gender','city']])

x_train_gender_city.shape


# In[30]:


# now lets concatenate everything

# butlets first differntiate age as well 

x_train_age = x_train.drop(columns = ['gender','fever','cough','city']).values

x_test_age  = x_test.drop(columns = ['gender','fever','cough','city']).values

x_train_age.shape


# In[35]:


x_train_Transformed = np.concatenate((x_train_age,x_train_fever,x_train_gender_city, x_train_cough), axis = 1)

x_test_Transformed = np.concatenate((x_test_age,x_test_fever,x_test_gender_city, x_test_cough), axis = 1)

x_train_Transformed.shape


# In[36]:


# Check the shapes of all arrays to identify the issue
print("x_train_age shape:", x_train_age.shape)
print("x_train_fever shape:", x_train_fever.shape)
print("x_train_gender_city shape:", x_train_gender_city.shape)
print("x_train_cough shape:", x_train_cough.shape)

print("\nx_test_age shape:", x_test_age.shape)
print("x_test_fever shape:", x_test_fever.shape)
print("x_test_gender_city shape:", x_test_gender_city.shape)
print("x_test_cough shape:", x_test_cough.shape)


# In[ ]:


# everything took so much time too redundant


# ##  Mentos Zindagi

# In[37]:


from sklearn.compose import ColumnTransformer



# In[41]:


transformer = ColumnTransformer(transformers = [
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2', OrdinalEncoder(categories = [['Mild','Strong']]),['cough']),
    ('tnf3', OneHotEncoder(sparse_output = False,drop = 'first'),['gender','city'])
], remainder = 'passthrough')


# In[45]:


transformer.fit_transform(x_train).shape


# In[46]:


transformer.transform(x_test).shape


# In[ ]:




