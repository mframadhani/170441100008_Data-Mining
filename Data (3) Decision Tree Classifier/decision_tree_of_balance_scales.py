#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np


# In[4]:


data = pd.read_csv('balance_scale.csv')
data.head()


# In[5]:


data.info()


# In[6]:


zero_not_accepted = ['berat_kiri','jarak_kiri','berat_kanan','jarak_kanan']
# for col in zero_not_accepted:
#     for i in data[col]:
#         if i==0:
#             colSum = sum(data[col])
#             meanCol=colSum/len(data[col])
#             data[col]=meanCol

for col in zero_not_accepted:
    data[col]= data[col].replace(0,np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN,mean)


# In[8]:


X = data.iloc[:,0:3]
y = data.iloc[:,3]
#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age
#build model & train data
X = data[['berat_kiri','jarak_kiri','berat_kanan','jarak_kanan']]
y = data['seimbang']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


# In[13]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[14]:


feature_cols = ['berat_kiri','jarak_kiri','berat_kanan','jarak_kanan']
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['B','R','L'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('keseimbangan.png')
Image(graph.create_png())


# In[ ]:




