#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:12:37 2019
@author: Anil Onal
The file cleans...
"""
import pandas as pd
import os

def select_feats():
    """
    This function selects the relevant features and the y variable.
    It also recodes some values for continuous/ordinal variables.
    """
    path = '/Users/flatironschol/Blogs/Blog-3/parental-involvement-to-education'
    os.chdir(path)
    parental_involvement_df = pd.read_csv('./data/pfi_pu.csv')
    parental_involvement_df = parental_involvement_df.loc[parental_involvement_df.HOMESCHLX == 2]
    cols = list(parental_involvement_df.columns)
    y_labels = cols[14:23] + cols[36:39] + [cols[40]]
    school_characteristics_labels = cols[4:13] + [cols[41]] + cols[61:76] + \
                                    [cols[214]] + [cols[218]] + [cols[244]] + [cols[366]]
    parent_characteristics_labels = [cols[76]] + [cols[241]] + [cols[243]] + cols[245:258] + \
                                    [cols[259]] + cols[264:331] + [cols[362]]  
    student_characteristics_labels = [cols[3]] + [cols[77]] + cols[199:214] + \
                                     cols[224:229] + cols[233:241] + [cols[333]]
    parental_involvement_labels = [cols[39]] + cols[52:61] + cols[78:81] + cols[184:199] + [cols[219]]
    X_cont_labels = [cols[60]] + cols[64:76] + cols[79:81] + [cols[191]] + [cols[199]] + \
                    cols[245:258] + [cols[274]] + [cols[282]] + [cols[285]] + cols[287:290] + \
                    [cols[302]] + [cols[310]] + [cols[313]] + cols[315:318] + cols[326:328] + \
                    [cols[362]] + [cols[366]]
    parental_involvement_df.loc[(parental_involvement_df.FHHOME == 5) | 
                                (parental_involvement_df.FHHOME == 6), 'FHHOME'] = 0
    parental_involvement_df.loc[parental_involvement_df.FHWKHRS == -1, 'FHWKHRS'] = 0
    parental_involvement_df.loc[parental_involvement_df.FHCHECKX == -1, 'FHCHECKX'] = 0
    parental_involvement_df.loc[(parental_involvement_df.FHHELP == 5) | 
                                (parental_involvement_df.FHHELP == -1), 'FHHELP'] = 0
    parental_involvement_df.loc[parental_involvement_df.P1HRSWK == -1, 'P1HRSWK'] = 0
    parental_involvement_df.loc[parental_involvement_df.P2HRSWK == -1, 'P2HRSWK'] = 0
    parental_involvement_df.loc[parental_involvement_df.P2MTHSWRK == -1, 'P2MTHSWRK'] = 0
    parental_involvement_df.loc[parental_involvement_df.P1MTHSWRK == -1, 'P1MTHSWRK'] = 0
    parental_involvement_df = parental_involvement_df.loc[parental_involvement_df.S16NUMST > 0]
    parental_involvement_df = parental_involvement_df.loc[parental_involvement_df.P1AGEPAR > 0]
    parental_involvement_df = parental_involvement_df.loc[parental_involvement_df.P1AGE > 0]
    y_df = parental_involvement_df[y_labels]
    school_characteristics_df = parental_involvement_df[school_characteristics_labels]
    parent_characteristics_df = parental_involvement_df[parent_characteristics_labels]
    parent_characteristics_df = parent_characteristics_df.drop(columns = ['P2AGE', 'P2AGEPAR', 'P1AGEMV', 'P2AGEMV'])
    X_cont_labels.remove('P2AGEPAR')
    X_cont_labels.remove('P2AGE')
    X_cont_labels.remove('P1AGEMV')
    X_cont_labels.remove('P2AGEMV')
    student_characteristics_df = parental_involvement_df[student_characteristics_labels]
    parental_involvement_df = parental_involvement_df[parental_involvement_labels]
    return y_df, \
           school_characteristics_df, \
           parent_characteristics_df, \
           student_characteristics_df, \
           parental_involvement_df, \
           X_cont_labels

def 