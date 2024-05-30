#!/usr/bin/env python
# coding: utf-8

# In[73]:


#################################################################
# Import libraries
#################################################################
import random
import string

# Modelling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
# import graphviz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import sys
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qboost.qboost import QBoostClassifier
from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dwave.system.composites import EmbeddingComposite
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics

# import psycopg2
# from psycopg2.extras import RealDictCursor
from imblearn.under_sampling import RandomUnderSampler
from timeit import default_timer as timer
from datetime import *
from collections import Counter
from connect import connect

# Garbage collection
import gc

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


# In[2]:


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)


# In[3]:


import seaborn as sns


# In[46]:


gc.collect()


# In[5]:


def loadData(path):
    data = pd.read_csv(path)
    # print(data)
    return data


# In[6]:


path = '../../../Dataset/IoT Intrusion Dataset/IoT Network Intrusion Dataset.csv'


# In[7]:


dataframe =  loadData(path)


# In[8]:


df = dataframe


# ### Data cleaning
# #### 1. Remove all columns which have more than 75% data missing. Check against NaN values.
# #### 2. Remove all columns which have (-)ve correlation with the target value
# #### 3. Remove all columns which have 0 correlation
# #### 4. Remove all columns which have corellation at the 1000th position

# #### Find non-numeric features

# In[9]:


non_numeric_columns = df.select_dtypes(exclude=[int, float]).columns.tolist()
print(non_numeric_columns)


# In[10]:


dataframe.head()


# In[11]:


df.head()


# In[12]:


df['Label'].unique()


# In[13]:


df['Label'].value_counts()['Normal']


# In[14]:


df['Label'].value_counts()['Anomaly']


# #### Remove the Target Variable / Dependent Variable from the non-numeric columns list

# In[15]:


# Label is non numeric but it is the dependent variable; 
# need for future data processing, remove it from the non numeric column list
non_numeric_columns.remove('Label')


# In[16]:


non_numeric_columns


# #### Remove non numeric columns from the dataset, but keeping the 'Label'

# In[17]:


df = df.drop(columns=non_numeric_columns)


# #### Change "Label" from text to numbers

# In[18]:


df['Label'] = df['Label'].replace('Anomaly', -1).replace('Normal', 1)


# #### Find out the datatypes of the columns in the dataframe

# In[19]:


column_info = df.dtypes
print(column_info)


# #### Find out the columns which have NaN / Missing values - None here

# In[20]:


nan_values = df.columns[df.isna().any()]


# In[21]:


nan_values


# #### Find out the columns where all values are equal -> does not contribute to the result

# In[22]:


columns_with_same_val = df.columns[(df == df.iloc[0]).all()]


# In[23]:


columns_with_same_val


# In[24]:


df = df.drop(columns=columns_with_same_val)


# In[25]:


df.head()


# #### Find out the max values from each column to avoid 'inf' -> infinity value problem

# In[26]:


df.max(axis=0)


# ##### Some columns above have Infinity value and cannot be processed as it is, so they need to be replaced with NaN and then imputed

# #### Replacing Infinity values with NaN

# In[27]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# #### Imputing using median for outliers

# In[28]:


df.fillna(df.median(), inplace=True)


# #### Separate Dependent and Independent columns

# In[29]:


shp = df.shape
cols = list(df.columns.values)


# In[30]:


X = df[cols[0:shp[1]-1]]


# In[67]:


y = df['Label']


# In[32]:


X.head(), y.head()


# ### Feature selection using PCA

# #### Clear memory by clearing variables

# In[33]:


# %reset


# #### Preprocess the data by scaling the features

# In[34]:


# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Use MinMaxScaler
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)


# In[35]:


scaled_X


# #### Perform PCA

# In[36]:


pca = PCA()
pca.fit(scaled_X)


# In[37]:


explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()


# #### Finding the best number of components by Elbow method

# In[38]:


# Calculate the difference in explained variance between components
explained_variance_diff = np.diff(explained_variance)

# Find the index of the elbow point (maximum difference)
elbow_index = np.argmax(explained_variance_diff) + 1

print("Number of components at the elbow point:", elbow_index)


# #### Find the best number of components by the Threshold method

# In[39]:


threshold =0.9985  # Define the desired threshold (e.g., 99% variance explained)

# Find the number of components above the threshold
n_components = np.argmax(cumulative_variance >= threshold) + 1

# Plot the threshold line
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=threshold, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()

print("Number of components above the threshold:", n_components)


# #### From the above mentioned statistical analysis, we see that including more or less 20 components is enough for a variance that levels off

# In[52]:


n_components = 20  # Choose the desired number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(scaled_X)


# #### Number of Components with impactful positive correlation = 20

# In[68]:


y_pca = y


# In[91]:


X_pca, y_pca


# In[ ]:


# for n_components in range(1,25):
#    pca = PCA(n_components=n_components)
#    X_pca = pca.fit_transform(scaled_X)


# In[ ]:


# Delete variable
del val


# #### The part below is for checking the recall based on the the number of k neighbors used in SMOTE oversampling

# In[49]:


# values to evaluate
k_values = [1, 2, 3, 4, 5, 6, 7]
for k in k_values:
    # define pipeline
    model = DecisionTreeClassifier()
    over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X_pca, y, scoring='recall', cv=cv, n_jobs=-1)
    score = mean(scores)
    print('> k=%d, Mean Recall: %.3f' % (k, score))


# #### The code below uses the QBoost to find out good parameters for oversampling, and undersampling

# In[81]:


def trainQboost(x, y):
    # print('Training QBOOST Model... ')
    qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    start = timer()
    qboost.fit(x, y, emb_sampler, lmd=lmd, **DW_PARAMS)
    end = timer()
    train_time = end - start
    # print('QBoost training time in seconds :', train_time)
    return qboost, train_time


# In[82]:


def predictModel(model, x):
    # print('Prediction on model :', model)
    start = timer()
    predictions = model.predict(x)
    end = timer()
    predict_time = end - start
    return predictions, predict_time


# In[80]:


DW_PARAMS = {'num_reads': 3000,
             'auto_scale': True,
             # "answer_mode": "histogram",
             'num_spin_reversal_transforms': 10,
             # 'annealing_time': 10,
             # 'postprocess': 'optimization',
             }
NUM_WEAK_CLASSIFIERS = 35
TREE_DEPTH = 3
dwave_sampler = DWaveSampler(token="DEV-98f903479d1e03bc59d7ba92376a492f76f7c906")
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.5


# In[2]:


y_pca


# In[93]:


def evaluate_over_under():
    # Nearest neighbor values for SMOTE
    k_values = [1, 2, 3, 4, 5, 6, 7]
    s_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Create a KFold object with n_splits=10
    # kf = KFold(n_splits=10)
    # precision, recall, f1 = 0, 0, 0
    for s in s_values:
        for k in k_values:
            # Declear the models for SMOTE over and Random Undersampling
            over = SMOTE(sampling_strategy=s, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            X, y = over.fit_resample(X_pca, y_pca)
            X, y = under.fit_resample(X, y)
            print(Counter(y))
            count = 0
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
            model, train_time = trainQboost(x_train, y_train)
            predictions, predict_time = predictModel(model, x_test)
            # print('Training Time: %.2f, Prediction Time: %2f' % (train_time, predict_time))
            # Calculate precision
            precision = precision_score(y_test, predictions)
            # Calculate recall
            recall = recall_score(y_test, predictions)
            # Calculate F1-score
            f1 = f1_score(y_test, predictions)
            print('>> s: %3f -> k: %d -> Precision: %.3f, Recall: %.3f, F1_score: %.3f, Training Time: %.2f, Prediction Time: %2f' % 
                  (s, k, precision, recall, f1, train_time, predict_time))
            
            # Iterate over the splits for 10 fold validation
            # for train_index, test_index in kf.split(X, y):
            #     # Obtain the training and testing subsets for the current split
            #     # X_train, X_test = X[train_index], X[test_index]
            #     # y_train, y_test = y[train_index], y[test_index]
            #     model, train_time = trainQboost(X[train_index], y[train_index])
            #     predictions, predict_time = predictModel(model, X[test_index])
            #     print('Training Time: %.2f, Prediction Time: %2f' % (train_time, predict_time))
            #     # Calculate precision
            #     precision = precision + precision_score(y[test_index], predictions)
            #     print('Precision: %.3f' % precision)
            #     # Calculate recall
            #     print('Precision: %.3f' % precision)
            #     recall = recall + recall_score(y[test_index], predictions)
            #     # Calculate F1-score
            #     f1 = f1 + f1_score(y[test_index], predictions)
            #     count = count + 1 
            # print('> k=%d, Mean Precision: %.3f, Recall: %.3f, F1-Score: %3f' % (k, precision/count, recall/count, f1/count))    
        


# In[94]:


evaluate_over_under()


# In[88]:


print(Counter(y_pca))


# In[77]:


(40073/585710)*100


# In[1]:


(292855/585710)*100


# In[ ]:





# #### Apply QBoost

# In[139]:


def reportResult(y, predictions, model, predict_type, train_time, predict_time):
    print("Reporting Result ...")
    result_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report = classification_report(y, predictions)
    with open('../results/' + str(result_time) + '_' + str(model) + '_' + str(predict_type) + '_report.txt', 'a') as f:
        f.write('Training time :' + str(train_time) + '\n')
        f.write(report)
        f.write('Prediction time :' + str(predict_time) + '\n')
    f.close()
    print("Saved...")


# In[123]:


x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, shuffle=True, random_state=32)


# In[141]:


model, train_time = trainQboost(x_train, y_train)


# In[142]:


predictions, predict_time = predictModel(model, x_train)


# In[143]:


reportResult(y_train, predictions, str(model)[1:4], 'training', train_time, predict_time)


# In[144]:


predictions, predict_time = predictModel(model, x_test)


# In[145]:


reportResult(y_test, predictions, str(model)[1:4], 'test', train_time, predict_time)


# In[43]:





# In[51]:


print(sys.getsizeof(df))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## The part below is for scratch work

# ### Correlation Matrix study

# In[ ]:


correlation_matrix = df.corr()


# In[ ]:


correlation_matrix


# In[ ]:


cm = correlation_matrix


# In[ ]:


cm


# In[ ]:


# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()


# In[ ]:


single_feature_correlation = correlation_matrix['Label']


# In[ ]:


single_feature_correlation


# In[ ]:


# Select NaN values
nan_values = single_feature_correlation[single_feature_correlation.isnull()]


# In[ ]:


nan_values


# In[ ]:


X['Flow_Byts/s']

