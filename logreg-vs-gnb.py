#!/usr/bin/env python
# coding: utf-8

# In[127]:


import numpy as np              
import matplotlib.pyplot as plt

mu, sigma = 0, 1 # mean and standard deviation
X1=np.random.normal(mu, sigma, 100)

mu, sigma = 1, 1 # mean and standard deviation
X2 = np.random.normal(mu, sigma, 100)

plt.scatter(X1,X2, s=50)
plt.show()


C1=np.vstack((X1, X2)).T

y_C1 = np.zeros(C1.shape[0])

mu, sigma = 0, 1 # mean and standard deviation
X1=np.random.normal(mu, sigma, 100)

mu, sigma = 1, 1 # mean and standard deviation
X2 = np.random.normal(mu, sigma, 100)

plt.scatter(X1,X2, s=50, c='y')
plt.show()

C2=np.vstack((X1, X2)).T

y_C2 = np.ones(C2.shape[0])

C2=np.vstack((X1, X2)).T

X= np.vstack((C1,C2))
y= np.vstack((np.array([y_C1]).T, np.array([y_C2]).T))


# In[128]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30) 


# In[129]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf1 = LogisticRegression()

clf1.fit(X_train,y_train.ravel())
y_pred_test1= clf1.predict(X_test)
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test1))  


# In[130]:


from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(X_train, y_train.ravel())
y_pred_test2 = clf2.predict(X_test)
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test2))  


# In[ ]:


# The accuracy is decreased significantly when the mean and
# standard deviation if C1 and C2 are the same.

