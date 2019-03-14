#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

wine = datasets.load_wine()

print(wine.feature_names)


# In[11]:


print(wine.target_names)


# In[12]:


X, Y = load_wine(return_X_y=True)     


# In[13]:


#X and Y are of type numpy.ndarray of float64. The dimensions 
#of X:(178, 13) and Y:(178,).                               

print("X type:",type(X),X.dtype,"X dim:",X.shape)
print("Y type:",type(Y),X.dtype,"Y dim:",Y.shape)         


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, 
                                                    random_state=1)    
#When you set the random_state parameter to a particular value, 
#the train-test split will be the same everytime you run it.


# In[15]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)               
y_pred_test = model.predict(X_test)


# In[16]:


print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test))       
print('Accuracy of train: %.2f' %accuracy_score(y_train, y_pred_train)) 

# The training accuracy is better, as the model was bulit using training data. 
# So this model will correctly classify the training data all the time.


# In[17]:


tree = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=1.0, max_features=4, bootstrap=True,
bootstrap_features=False, n_jobs=1, random_state=1)


# In[18]:


tree.fit(X_train,y_train)
y_pred_train_bag = tree.predict(X_train)               
y_pred_test_bag = tree.predict(X_test)
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test_bag))       
print('Accuracy of train: %.2f' %accuracy_score(y_train, y_pred_train_bag)) 

# On executing the code multiple times, the bagged classifier always worked as well as or better than
# the not bagged case. So the accuracy is iproved by bagging.


# In[ ]:




