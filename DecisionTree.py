#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn import datasets

wine = datasets.load_wine()      #1

print(wine.feature_names)       #2


# In[23]:


print(wine.target_names)         #3


# In[24]:


from sklearn.datasets import load_wine
X, Y = load_wine(return_X_y=True)                         #4

#X and Y are of type numpy.ndarray of float64. The dimensions 
#of X:(178, 13) and Y:(178,).                                #5

print("X type:",type(X),X.dtype,"X dim:",X.shape)
print("Y type:",type(Y),X.dtype,"Y dim:",Y.shape)         #6


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, 
                                                    random_state=1)    

#When you set the random_state parameter to a particular value, 
#the train-test split will be the same everytime you run it.   #7


# In[26]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)               #8
y_pred_test = model.predict(X_test)


# In[27]:


from sklearn.metrics import accuracy_score
print('Accuracy of test: %.2f' %accuracy_score(y_test, y_pred_test))       #9
print('Accuracy of train: %.2f' %accuracy_score(y_train, y_pred_train))    #10       


# In[28]:


# The training accuracy is better, as the model was bulit using training data. 
# so this model will correctly classify the training data all the time        #11

