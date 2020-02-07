#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:17:25 2019

@author: flatironschol
"""
"""
Created on Fri Nov 22 22:12:37 2019
@author: Anil Onal
The file cleans...
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.sparse
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import classification_report
df = pd.read_csv('/Users/flatironschol/Blogs/Blog-3/parental-involvement-to-education/data/pfi_pu.csv')
cols = list(df.columns)
subset = cols[52:67]
subset.extend(cols[69:81])
subset.extend(cols[184:199])
subset.append(cols[15])
pi_df = df.copy()
pi_df = pi_df[subset]
pi_df.describe().T
for c in pi_df.columns:
    pi_df = pi_df.loc[pi_df[c] != -1]
pi_df = pi_df.loc[pi_df.SEGRADES != 5]
pi_df.describe().T
y = pi_df.SEGRADES.values
X_cont = pi_df[['FSFREQ', 'FHWKHRS', 'FODINNERX']].values
X_cat = pi_df.drop(['SEGRADES', 'FSFREQ', 'FHWKHRS', 'FODINNERX'], axis = 1).values
X = np.concatenate((X_cont, X_cat), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    stratify = y, \
                                                    test_size = 0.2, \
                                                    random_state = 112219)
X_train_cont = X_train[:,0:3]
X_train_cat = X_train[:,3:]
X_test_cont = X_test[:,0:3]
X_test_cat = X_test[:,3:]
encdr = OneHotEncoder(handle_unknown = 'ignore')
X_train_encoded = encdr.fit_transform(X_train_cat)
X_train_encoded = X_train_encoded.toarray()
sclr = StandardScaler()
X_train_scaled = sclr.fit_transform(X_train_cont)
X_train = np.concatenate((X_train_scaled, X_train_encoded), axis = 1)
c_range = np.linspace(0.01, 2, 100)
accuracy = []
for c in c_range:
    clf = LogisticRegression(random_state = 112219, penalty='l1', solver='saga', \
                             multi_class='multinomial', max_iter = 2000, C = c)
    rslt = clf.fit(X_train, y_train)
    y_train_hat = rslt.predict(X_train)
    residuals = np.abs(y_train - y_train_hat)
    accuracy.append(pd.Series(residuals).value_counts(normalize=True)[0])
ax = sns.lineplot(c_range, accuracy, color = 'dodgerblue')
ax.set_title('Accuracy Score for Different C Values')
ax.set_xlabel('C values')
ax.set_ylabel('Accuracy')
clf = LogisticRegression(random_state=0, penalty='l1', solver='saga', \
                         multi_class='multinomial', max_iter = 2000, C = 0.3)
rslt = clf.fit(X_train, y_train)
y_train_hat = rslt.predict(X_train)
residuals = np.abs(y_train - y_train_hat)
beta = rslt.coef_
pd.Series(residuals).value_counts(normalize=True)
print(classification_report(y_train, y_train_hat))
import matplotlib.pyplot as plt
ax1 = sns.distplot(y_train, bins = 4, kde = False, color = 'dodgerblue', label = 'actual', hist_kws = {'alpha': 0.9})
sns.distplot(y_train_hat, bins = 4, kde = False, color = 'orange', label = 'prediction', hist_kws = {'alpha': 0.9}, ax = ax1)
ax1.set_title('Model Predictions for the Training Data')
ax1.set_ylabel('Number of students')
ax1.set_xlabel('Student grades')
plt.legend()

X_test_scaled = sclr.transform(X_test_cont)
X_test_encoded = encdr.transform(X_test_cat)
X_test_encoded = X_test_encoded.toarray()
X_test = np.concatenate((X_test_scaled, X_test_encoded), axis = 1)
y_test_hat = rslt.predict(X_test)
pd.Series(y_test_hat).value_counts()
residuals = np.abs(y_test - y_test_hat)
pd.Series(residuals).value_counts(normalize=True)
print(classification_report(y_test, y_test_hat))
ax1 = sns.distplot(y_test, bins = 4, kde = False, color = 'dodgerblue', label = 'actual', hist_kws = {'alpha': 0.9})
sns.distplot(y_test_hat, bins = 4, kde = False, color = 'orange', label = 'prediction', hist_kws = {'alpha': 0.9}, ax = ax1)
ax1.set_title('Model Predictions for the Test Data')
ax1.set_ylabel('Number of students')
ax1.set_xlabel('Student grades')
plt.legend()
