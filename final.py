#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
import lightgbm as lgb


# In[ ]:


import pandas as pd
train = pd.read_csv('train.csv', sep=',|;')
train.reset_index(inplace=True)
train = train.rename(columns={"index": "V1"})

feature_matrix = pd.read_csv('feamat.csv', sep=';|,')

train_data = train.merge(feature_matrix, on = ['V1'],how = 'left')


# In[ ]:


train_data = train_data.replace([np.inf, -np.inf], np.nan)


# In[ ]:


train_data = train_data.fillna(train_data.mean())


# In[ ]:


encoder = LabelEncoder()
feature_matrix['V1'] = encoder.fit_transform(feature_matrix['V1'])

train_data['V1'] = encoder.transform(train_data['V1'])


# In[ ]:


test = pd.read_csv('test.csv', sep=',|;')

test.reset_index(inplace=True)
test = test.rename(columns={"index": "V1"})

featest = pd.read_csv('feamat.csv', sep=';|,')
test_data =  test.merge(featest, on = ['V1'],how = 'left')

test_data = test_data.replace([np.inf, -np.inf], np.nan)
test_data = test_data.fillna(test_data.mean())

test_data['V1'] = encoder.transform(test_data['V1'])


# In[ ]:


constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(train_data)
train_data = pd.DataFrame(train_data)
len(train_data.columns[constant_filter.get_support()])

constant_columns = [column for column in train_data.columns
                    if column not in train_data.columns[constant_filter.get_support()]]
print(constant_columns)
train_data.drop(labels=constant_columns, axis=1, inplace=True)
test_data.drop(labels=constant_columns, axis=1, inplace=True)


# In[ ]:


qconstant_filter = VarianceThreshold(threshold=0.01)
qconstant_filter.fit(train_data)
len(train_data.columns[qconstant_filter.get_support()])
qconstant_columns = [column for column in train_data.columns
                    if column not in train_data.columns[qconstant_filter.get_support()]]
print(qconstant_columns)
train_data.drop(labels=qconstant_columns, axis=1, inplace=True)
test_data.drop(labels=qconstant_columns, axis=1, inplace=True)


# In[ ]:


train_features_T = train_data.T
train_features_T.shape
print(train_features_T.duplicated().sum())
unique_features = train_features_T.drop_duplicates(keep='first').T
unique_features.shape
duplicated_features = [dup_col for dup_col in train_data.columns if dup_col not in unique_features.columns]
print(duplicated_features)
train_data.drop(labels=duplicated_features, axis=1, inplace=True)
test_data.drop(labels=duplicated_features, axis=1, inplace=True)


# In[ ]:


correlated_features = set()
correlation_matrix = train_data.corr()
for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.90:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
len(correlated_features)
print(correlated_features)
train_data.drop(labels=correlated_features, axis=1, inplace=True)
test_data.drop(labels=correlated_features, axis=1, inplace=True)


# In[ ]:


X = train_data.drop(['Expected'], axis=1)
y = train_data['Expected']


# In[ ]:


fc = SelectKBest(f_classif, k=32)
best_f = fc.fit_transform(X, y)
best_f


# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# define pipeline
over = SMOTE(random_state=42)
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)


best_f, y = pipeline.fit_resample(best_f, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(best_f, y, test_size = 0.12, random_state = 42)


# In[ ]:


scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)


# In[ ]:


from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
# group / ensemble of models
estimator = []

estimator.append(('GB', GradientBoostingClassifier(n_estimators = 384, learning_rate = 0.05138021950826747, max_depth = 179, min_samples_leaf = 110, min_samples_split = 451, random_state = 42)))
estimator.append(('XG', XGBClassifier(n_estimators = 594, learning_rate = 0.015720655420875196, max_depth = 15, random_state = 42)))
estimator.append(('LGM', lgb.LGBMClassifier(n_estimators = 393, learning_rate = 0.03872954579312028, max_depth = 75, num_leaves = 242, random_state = 42)))

# Voting Classifier with hard voting


# clf = StackingClassifier(
# ...     estimators=estimators, final_estimator=LogisticRegression()
# ... )

vot_soft = VotingClassifier(estimators = estimator,  voting='hard')
vot_soft.fit(X_train, y_train)


# In[ ]:


print("Training Score {}".format(vot_soft.score(X_train, y_train)))
print("Testing Score {}".format(vot_soft.score(X_test, y_test)))
from sklearn.metrics import f1_score
y_pred = vot_soft.predict(X_test)
f1_score(y_test, y_pred, average='macro')


# In[ ]:


X_important = fc.transform(test_data)
#X_important = best_model.transform(df)

y_pred = vot_soft.predict(X_important)


# In[ ]:


testd = pd.read_csv('../input/toxic-xmen/test.csv', sep=',')
output = pd.DataFrame({'Id': testd["x"], 'Predicted': y_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

