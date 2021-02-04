#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[2]:


df_diabetes = pd.read_csv('diabetic_data_target.csv')


# In[3]:


df_diabetes.head()


# In[4]:


df_diabetes.tail()


# In[5]:


df_diabetes.info()


# In[6]:


df_diabetes.describe()


# In[7]:


df_diabetes.describe(include='all')


# In[8]:


df_diabetes.isnull().sum()


# In[9]:


df_diabetes = df_diabetes.replace('?', np.NaN)


# In[10]:


df_diabetes.isnull().sum()


# In[11]:


df_diabetes['race'] = df_diabetes['race'].fillna(df_diabetes['race'].mode)


# In[12]:


df_diabetes.isnull().sum()


# In[13]:


df_diabetes = df_diabetes.drop(['weight', 'payer_code', 'medical_specialty'], axis = 1)


# In[14]:


df_diabetes.isnull().sum()


# In[15]:


df_diabetes['readmitted'].value_counts()


# In[16]:


df_diabetes.info()


# In[17]:


df_diabetes.iloc[0]


# In[18]:


df_diabetes['number_emergency'].unique()


# In[19]:


df_diabetes = df_diabetes.drop(['encounter_id', 'patient_nbr', 'examide', 'citoglipton'], axis = 1)


# In[20]:


df_diabetes['race'] = df_diabetes.race.astype('string')
df_diabetes['race'] = df_diabetes.race.astype('category')
df_diabetes['gender'] = df_diabetes['gender'].astype('category')
df_diabetes['age'] = df_diabetes['age'].astype('category')
df_diabetes['A1Cresult'] = df_diabetes['A1Cresult'].astype('category')
df_diabetes['max_glu_serum'] = df_diabetes['max_glu_serum'].astype('category')
df_diabetes['metformin'] = df_diabetes['metformin'].astype('category')
df_diabetes['repaglinide'] = df_diabetes['repaglinide'].astype('category')
df_diabetes['nateglinide'] = df_diabetes['nateglinide'].astype('category')
df_diabetes['chlorpropamide'] = df_diabetes['chlorpropamide'].astype('category')
df_diabetes['glimepiride'] = df_diabetes['glimepiride'].astype('category')
df_diabetes['acetohexamide'] = df_diabetes['acetohexamide'].astype('category')
df_diabetes['glipizide'] = df_diabetes['glipizide'].astype('category')
df_diabetes['glyburide'] = df_diabetes['glyburide'].astype('category')
df_diabetes['tolbutamide'] = df_diabetes['tolbutamide'].astype('category')
df_diabetes['pioglitazone'] = df_diabetes['pioglitazone'].astype('category')
df_diabetes['rosiglitazone'] = df_diabetes['rosiglitazone'].astype('category')
df_diabetes['acarbose'] = df_diabetes['acarbose'].astype('category')
df_diabetes['miglitol'] = df_diabetes['miglitol'].astype('category')
df_diabetes['troglitazone'] = df_diabetes['troglitazone'].astype('category')
df_diabetes['tolazamide'] = df_diabetes['tolazamide'].astype('category')
df_diabetes['insulin'] = df_diabetes['insulin'].astype('category')
df_diabetes['glyburide-metformin'] = df_diabetes['glyburide-metformin'].astype('category')
df_diabetes['glipizide-metformin'] = df_diabetes['glipizide-metformin'].astype('category')
df_diabetes['glimepiride-pioglitazone'] = df_diabetes['glimepiride-pioglitazone'].astype('category')
df_diabetes['metformin-rosiglitazone'] = df_diabetes['metformin-rosiglitazone'].astype('category')
df_diabetes['metformin-pioglitazone'] = df_diabetes['metformin-pioglitazone'].astype('category')
df_diabetes['change'] = df_diabetes['change'].astype('category')
df_diabetes['diabetesMed'] = df_diabetes['diabetesMed'].astype('category')


# In[21]:


df_diabetes.info()


# In[22]:


df_diabetes['race'] = df_diabetes['race'].cat.codes
df_diabetes['gender'] = df_diabetes['gender'].cat.codes
df_diabetes['age'] = df_diabetes['age'].cat.codes
df_diabetes['A1Cresult'] = df_diabetes['A1Cresult'].cat.codes
df_diabetes['max_glu_serum'] = df_diabetes['max_glu_serum'].cat.codes
df_diabetes['metformin'] = df_diabetes['metformin'].cat.codes
df_diabetes['repaglinide'] = df_diabetes['repaglinide'].cat.codes
df_diabetes['nateglinide'] = df_diabetes['nateglinide'].cat.codes
df_diabetes['chlorpropamide'] = df_diabetes['chlorpropamide'].cat.codes
df_diabetes['glimepiride'] = df_diabetes['glimepiride'].cat.codes
df_diabetes['acetohexamide'] = df_diabetes['acetohexamide'].cat.codes
df_diabetes['glipizide'] = df_diabetes['glipizide'].cat.codes
df_diabetes['glyburide'] = df_diabetes['glyburide'].cat.codes
df_diabetes['tolbutamide'] = df_diabetes['tolbutamide'].cat.codes
df_diabetes['pioglitazone'] = df_diabetes['pioglitazone'].cat.codes
df_diabetes['rosiglitazone'] = df_diabetes['rosiglitazone'].cat.codes
df_diabetes['acarbose'] = df_diabetes['acarbose'].cat.codes
df_diabetes['miglitol'] = df_diabetes['miglitol'].cat.codes
df_diabetes['troglitazone'] = df_diabetes['troglitazone'].cat.codes
df_diabetes['tolazamide'] = df_diabetes['tolazamide'].cat.codes
df_diabetes['insulin'] = df_diabetes['insulin'].cat.codes
df_diabetes['glyburide-metformin'] = df_diabetes['glyburide-metformin'].cat.codes
df_diabetes['glipizide-metformin'] = df_diabetes['glipizide-metformin'].cat.codes
df_diabetes['glimepiride-pioglitazone'] = df_diabetes['glimepiride-pioglitazone'].cat.codes
df_diabetes['metformin-rosiglitazone'] = df_diabetes['metformin-rosiglitazone'].cat.codes
df_diabetes['metformin-pioglitazone'] = df_diabetes['metformin-pioglitazone'].cat.codes
df_diabetes['change'] = df_diabetes['change'].cat.codes
df_diabetes['diabetesMed'] = df_diabetes['diabetesMed'].cat.codes


# In[23]:


df_diabetes_features = df_diabetes.drop('readmitted', axis = 1)


# In[24]:


df_diabetes_target = df_diabetes['readmitted']


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_diabetes_features, df_diabetes_target, train_size = 0.7)


# In[26]:


# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()


# # In[27]:


# lr.fit(x_train, y_train)


# # In[28]:


# lr.score(x_train, y_train)


# # In[29]:


# y_pred = lr.predict(x_test)


# # In[30]:


from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_pred)


# In[31]:


# from sklearn.tree import DecisionTreeClassifier
# 
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)
# print(dt.score(x_train, y_train))
# accuracy_score(dt.predict(x_test), y_test)





# In[32]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, criterion = 'entropy', max_depth = 50)
rf.fit(x_train, y_train)
print(rf.score(x_train, y_train))
accuracy_score(rf.predict(x_test), y_test)
pkl_filename = 'model/model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(rf, open(pkl_filename, 'wb'))

# In[33]:


# import tensorflow as tf
# from tensorflow import keras


# # In[46]:


# model = keras.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(40, activation='relu'),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])


# # In[53]:


# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# # In[54]:


# history = model.fit(x_train.values, y_train.values, epochs=10, batch_size=10)


# # In[55]:


# test_loss, test_acc = model.evaluate(x_test.values,  y_test.values, verbose = 2)

# print('\nTest accuracy:', test_acc)


# # In[56]:


# print(history.history.keys())


# # In[57]:


# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# # In[58]:


# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# In[ ]:




