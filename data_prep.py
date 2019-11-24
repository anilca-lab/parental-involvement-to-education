#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:12:37 2019
@author: Anil Onal
The file cleans...
"""
import pandas as pd
df = pd.read_csv('/Users/flatironschol/Blogs/Blog-3/parental-involvement-to-education/data/pfi_pu.csv')
cols = list(df.columns)
subset = cols[52:81]
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
X_cont = pi_df[['FSFREQ', 'FHWKHRS']].values
X_cat = pi_df.drop(['SEGRADES', 'FSFREQ', 'FHWKHRS'], axis = 1).values
from sklearn.preprocessing import OneHotEncoder
encdr = OneHotEncoder()
X_encoded = encdr.fit_transform(X_cat)
import scipy.sparse
X_encoded = X_encoded.toarray()
from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
X_scaled = sclr.fit_transform(X_cont)
import numpy as np
X = np.concatenate((X_scaled, X_cat), axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 112219)
from sklearn.linear_model import Lasso
l = Lasso(alpha = 1)
rslt = l.fit(X_train, y_train)
y_train_hat = rslt.predict(X_train)
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_train, y_train_hat)
mean_squared_error(y_train, y_train_hat)
beta = rslt.coef_
from sklearn.linear_model import Ridge
r = Ridge(alpha = 1)
rslt = r.fit(X_train, y_train)
y_train_hat = rslt.predict(X_train)
r2_score(y_train, y_train_hat)
mean_squared_error(y_train, y_train_hat)
beta = rslt.coef_
rslt = clf.fit(X, y)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, penalty='l1', solver='saga', multi_class='multinomial', max_iter = 2000)
rslt = clf.fit(X, y)
y_train_hat = rslt.predict(X_train)
r2_score(y_train, y_train_hat)
mean_squared_error(y_train, y_train_hat)
beta = rslt.coef_
import seaborn as sns
sns.distplot(y_train, bins = 4, kde = False)
sns.distplot(y_train_hat, bins = 4, kde = False)