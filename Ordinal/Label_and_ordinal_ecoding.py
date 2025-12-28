#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as mp


# In[14]:


df = pd.read_csv('customer.csv')


# In[3]:


df.head()


# In[ ]:





# In[15]:


df = df.iloc[:,2:]


# In[16]:


df.head()


# In[ ]:





# In[4]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('purchased', axis=1),
                                                    df['purchased'],
                                                    test_size=0.3,
                                                    random_state=0)
X_train.shape, X_test.shape


# In[22]:


from sklearn.preprocessing import OrdinalEncoder


# In[27]:


oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])


# In[28]:


oe.fit(X_train)


# In[29]:


X_train = oe.transform(X_train)
X_test = oe.transform(X_test)


# In[30]:


X_train


# In[34]:


from sklearn.preprocessing import LabelEncoder


# In[35]:


Lb = LabelEncoder()


# In[36]:


Lb.fit(y_train)


# In[37]:


y_train = Lb.transform(y_train)
y_test = Lb.transform(y_test)


# In[38]:


y_test


# In[40]:


Lb.classes_


# In[41]:


from sklearn.metrics import accuracy_score


# In[ ]:





# In[42]:


y_pred = Lb.predict(y_test)


# In[ ]:




