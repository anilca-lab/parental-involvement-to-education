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
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to transform categorical variables
def encoder_transform(encoder, X):
    X_encoded = encoder.transform(X).toarray()
    encoded_feats = list(encoder.get_feature_names())
    feats = X.columns
    encoded_feats_updated = []
    for feat in encoded_feats:
        feat_split = feat.split('_')
        i = int(feat_split[0][1:])
        dummies = feat_split[1]
        feat_updated = f'{feats[i]}_{dummies}'
        encoded_feats_updated.append(feat_updated)
    return pd.DataFrame(X_encoded, columns = encoded_feats_updated)

def calculate_vif_(X_df, thresh = 5):
    feats = X_df.columns
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X_df[[feats]].values, ix)
               for ix in feats]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + feats[maxloc] +
                  '\' at index: ' + str(maxloc))
            del feats[maxloc]
            dropped = True
    print('Remaining variables:')
    print(X_df.columns)
    return X[[feats]]