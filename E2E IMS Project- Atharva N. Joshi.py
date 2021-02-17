#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


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


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df_diabetes['race'])


# In[9]:


sns.countplot(df_diabetes['gender'])


# In[10]:


sns.countplot(df_diabetes['max_glu_serum'])


# In[11]:


sns.countplot(df_diabetes['A1Cresult'])


# In[12]:


sns.countplot(df_diabetes['metformin'])


# In[13]:


sns.countplot(df_diabetes['repaglinide'])


# In[14]:


sns.countplot(df_diabetes['nateglinide'])


# In[15]:


sns.countplot(df_diabetes['chlorpropamide'])


# In[16]:


sns.countplot(df_diabetes['glimepiride'])


# In[17]:


sns.countplot(df_diabetes['acetohexamide'])


# In[18]:


sns.countplot(df_diabetes['glipizide'])


# In[19]:


sns.countplot(df_diabetes['glyburide'])


# In[20]:


sns.countplot(df_diabetes['tolbutamide'])


# In[21]:


sns.countplot(df_diabetes['pioglitazone'])


# In[22]:


sns.countplot(df_diabetes['rosiglitazone'])


# In[23]:


sns.countplot(df_diabetes['acarbose'])


# In[24]:


sns.countplot(df_diabetes['miglitol'])


# In[25]:


sns.countplot(df_diabetes['troglitazone'])


# In[26]:


sns.countplot(df_diabetes['tolazamide'])


# In[27]:


sns.countplot(df_diabetes['examide'])


# In[28]:


sns.countplot(df_diabetes['citoglipton'])


# In[29]:


sns.countplot(df_diabetes['insulin'])


# In[30]:


sns.countplot(df_diabetes['glyburide-metformin'])


# In[31]:


sns.countplot(df_diabetes['glipizide-metformin'])


# In[32]:


sns.countplot(df_diabetes['glimepiride-pioglitazone'])


# In[33]:


sns.countplot(df_diabetes['metformin-rosiglitazone'])


# In[34]:


sns.countplot(df_diabetes['metformin-pioglitazone'])


# In[35]:


sns.countplot(df_diabetes['change'])


# In[36]:


sns.countplot(df_diabetes['diabetesMed'])


# In[37]:


sns.countplot(df_diabetes['readmitted'])


# In[38]:


sns.lineplot(df_diabetes['age'],df_diabetes['time_in_hospital'])


# In[39]:


sns.lineplot(df_diabetes['age'],df_diabetes['num_lab_procedures'])


# In[40]:


sns.lineplot(df_diabetes['age'],df_diabetes['num_procedures'])


# In[41]:


sns.lineplot(df_diabetes['age'],df_diabetes['number_outpatient'])


# In[42]:


sns.lineplot(df_diabetes['age'],df_diabetes['number_inpatient'])


# In[43]:


sns.lineplot(df_diabetes['age'],df_diabetes['number_diagnoses'])


# In[44]:


df_diabetes.isnull().sum()


# In[45]:


df_diabetes = df_diabetes.replace('?', np.NaN)


# In[46]:


df_diabetes.isnull().sum()


# In[47]:


for i in df_diabetes.columns:
    print(i + ':' + str(df_diabetes[i].unique()))


# In[48]:


df_diabetes['race'] = df_diabetes['race'].fillna(df_diabetes['race'].mode)


# In[49]:


df_diabetes.isnull().sum()


# In[50]:


df_diabetes = df_diabetes.drop(['weight', 'payer_code', 'medical_specialty'], axis = 1)


# In[51]:


df_diabetes.isnull().sum()


# In[52]:


df_diabetes['gender'].value_counts()


# In[53]:


df_diabetes.info()


# In[54]:


df_diabetes.iloc[0]


# In[55]:


df_diabetes['max_glu_serum'].unique()


# In[56]:


df_diabetes = df_diabetes.drop(['encounter_id', 'patient_nbr', 'examide', 'citoglipton'], axis = 1)
#examide and citoglipton since it has only one value


# In[57]:


df_diabetes['race'] = df_diabetes.race.astype('string')
df_diabetes['race'] = df_diabetes.race.astype('category')
df_diabetes['gender'] = df_diabetes['gender'].astype('category')
df_diabetes['age'] = df_diabetes['age'].astype('category')
df_diabetes['admission_type_id'] = df_diabetes['admission_type_id'].astype('category')
df_diabetes['discharge_disposition_id'] = df_diabetes['discharge_disposition_id'].astype('category')
df_diabetes['admission_source_id'] = df_diabetes['admission_source_id'].astype('category')
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


# In[58]:


df_diabetes.info()


# In[59]:


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


# In[60]:


df_diabetes_features = df_diabetes.drop('readmitted', axis = 1)


# In[61]:


df_diabetes_target = df_diabetes['readmitted']


# In[62]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_diabetes_features, df_diabetes_target, train_size = 0.7)


# In[63]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[64]:


lr.fit(x_train, y_train)


# In[65]:


lr.score(x_train, y_train)


# In[66]:


y_pred = lr.predict(x_test)


# In[67]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[68]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print(dt.score(x_train, y_train))
accuracy_score(dt.predict(x_test), y_test)


# In[82]:


from sklearn.ensemble import RandomForestClassifier
import pickle
rf = RandomForestClassifier(n_estimators=200, criterion = 'entropy', max_depth = 50)
rf.fit(x_train, y_train)
print(rf.score(x_train, y_train))
print(accuracy_score(rf.predict(x_test), y_test))
pkl_filename = 'model/model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(rf, open(pkl_filename, 'wb'))


# In[70]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
bagging_clf = BaggingClassifier(rf, 
                                max_samples=0.4, max_features=10, random_state=0)
bagging_scores = cross_val_score(bagging_clf, x_train, y_train, cv=10, n_jobs=-1)


# In[71]:


print(bagging_scores)


# In[72]:


from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
ada_boost = AdaBoostClassifier()
grad_boost = GradientBoostingClassifier()
xgb_boost = XGBClassifier()
boost_array = [ada_boost, grad_boost, xgb_boost]
eclf = EnsembleVoteClassifier(clfs=[ada_boost, grad_boost, xgb_boost], voting='hard')
labels = ['Ada Boost', 'Grad Boost', 'XG Boost', 'Ensemble']
for clf, label in zip([ada_boost, grad_boost, xgb_boost, eclf], labels):
    scores = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    print("Mean: {0:.3f}, std: (+/-) {1:.3f} [{2}]".format(scores.mean(), scores.std(), label))


# In[73]:


import tensorflow as tf
from tensorflow import keras


# In[74]:


model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


# In[75]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[76]:


history = model.fit(x_train.values, y_train.values, epochs=10, batch_size=10)


# In[77]:


test_loss, test_acc = model.evaluate(x_test.values,  y_test.values, verbose = 2)

print('\nTest accuracy:', test_acc)


# In[78]:


print(history.history.keys())


# In[79]:


import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[80]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




