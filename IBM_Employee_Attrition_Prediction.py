#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from patsy import dmatrices
import sklearn
import seaborn as sns


# In[2]:


dataframe=pd.read_csv("IBM Attrition Data.csv")


# In[3]:


dataframe.head()


# In[4]:


names = dataframe.columns.values 
print(names)


# In[5]:


# histogram for age
plt.figure(figsize=(10,8))
dataframe['Age'].hist(bins=70)
plt.title("Age distribution of Employees")
plt.xlabel("Age")
plt.ylabel("# of Employees")
plt.show()


# In[6]:


# explore data for Attrition by Age
plt.figure(figsize=(14,10))
plt.scatter(dataframe.Attrition,dataframe.Age, alpha=.55)
plt.title("Attrition by Age ")
plt.ylabel("Age")
plt.grid(b=True, which='major',axis='y')
plt.show()


# In[7]:


# explore data for Left employees breakdown
plt.figure(figsize=(8,6))
dataframe.Attrition.value_counts().plot(kind='barh',color='blue',alpha=.65)
plt.title("Attrition breakdown ")
plt.show()


# In[8]:


# explore data for Education Field distribution
plt.figure(figsize=(10,8))
dataframe.EducationField.value_counts().plot(kind='barh',color='g',alpha=.65)
plt.title("Education Field Distribution")
plt.show()


# In[9]:


# explore data for Marital Status
plt.figure(figsize=(8,6))
dataframe.MaritalStatus.value_counts().plot(kind='bar',alpha=.5)
plt.show()


# In[10]:


dataframe.describe()


# In[11]:


dataframe.info()


# In[12]:


dataframe.columns


# In[13]:


dataframe.std()


# In[14]:


dataframe['Attrition'].value_counts()


# In[15]:


dataframe['Attrition'].dtypes


# In[16]:


dataframe['Attrition'].replace('Yes',1, inplace=True)
dataframe['Attrition'].replace('No',0, inplace=True)


# In[17]:


dataframe.head(10)


# In[18]:


# building up a logistic regression model
X = dataframe.drop(['Attrition'],axis=1)
X.head()
Y = dataframe['Attrition']
Y.head()


# In[19]:


dataframe['EducationField'].replace('Life Sciences',1, inplace=True)
dataframe['EducationField'].replace('Medical',2, inplace=True)
dataframe['EducationField'].replace('Marketing', 3, inplace=True)
dataframe['EducationField'].replace('Other',4, inplace=True)
dataframe['EducationField'].replace('Technical Degree',5, inplace=True)
dataframe['EducationField'].replace('Human Resources', 6, inplace=True)


# In[20]:


dataframe['EducationField'].value_counts()


# In[21]:


dataframe['Department'].value_counts()


# In[22]:


dataframe['Department'].replace('Research & Development',1, inplace=True)
dataframe['Department'].replace('Sales',2, inplace=True)
dataframe['Department'].replace('Human Resources', 3, inplace=True)


# In[23]:


dataframe['Department'].value_counts()


# In[24]:


dataframe['MaritalStatus'].value_counts()


# In[25]:


dataframe['MaritalStatus'].replace('Married',1, inplace=True)
dataframe['MaritalStatus'].replace('Single',2, inplace=True)
dataframe['MaritalStatus'].replace('Divorced',3, inplace=True)


# In[26]:


dataframe['MaritalStatus'].value_counts()


# In[27]:


x=dataframe.select_dtypes(include=['int64'])
x.dtypes


# In[28]:


x.columns


# In[29]:


y=dataframe['Attrition']


# In[30]:


y.head()


# In[31]:


y, x = dmatrices('Attrition ~ Age + Department +                   DistanceFromHome + Education + EducationField + YearsAtCompany',
                  dataframe, return_type="dataframe")
print (x.columns)


# In[32]:


y = np.ravel(y)


# In[33]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model = model.fit(x, y)

# check the accuracy on the training set
model.score(x, y)


# In[34]:


y.mean()


# In[35]:


X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y, test_size=0.3, random_state=0)
model2=LogisticRegression()
model2.fit(X_train, y_train)


# In[36]:


predicted= model2.predict(X_test)
print (predicted)


# In[37]:


probs = model2.predict_proba(X_test)
print (probs)


# In[38]:


from sklearn import metrics

print (metrics.accuracy_score(y_test, predicted))
print (metrics.roc_auc_score(y_test, probs[:, 1]))


# In[39]:


print (metrics.confusion_matrix(y_test, predicted))
print (metrics.classification_report(y_test, predicted))


# In[40]:


print (X_train)


# In[41]:


#add random values to KK according to the parameters mentioned above to check the proabily of attrition of the employee
kk=[[1.0, 23.0, 1.0, 500.0, 3.0, 24.0, 1.0]]
print(model.predict_proba(kk))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




