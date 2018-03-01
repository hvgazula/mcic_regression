#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:32:34 2018

@author: hgazula
"""

import pickle
import pandas as pd
import statsmodels.api as sm


def select_and_drop_cols(site_dummy, site_data):
    """Select and crop columns"""
    #    select_column_list = [
    #        'age', 'site_MGH', 'site_UMN', 'site_UNM', 'diagnosis', 'sex'
    #    ]
    select_column_list = ['age', 'diagnosis', 'sex']
    #    site_data = site_dummy.merge(site_data, on='site', how='right')
    site_data = site_data.drop('site', axis=1)
    site_X = site_data[select_column_list]
    site_y = site_data.drop(select_column_list, axis=1)
    return site_X, site_y


def get_dummies_and_augment(site_X):
    """Add a constant column and get dummies for categorical values"""
    X = pd.get_dummies(site_X, drop_first='True')
    X = sm.add_constant(X, has_constant='add')
    return X


def load_data():

    with open("final_data.pkl", "rb") as f:
        demographics, voxels = pickle.load(f)

    FinalData = pd.concat([demographics, voxels], axis=1)

    site_01 = FinalData[FinalData['site'].str.match('IA')]
    site_02 = FinalData[FinalData['site'].str.match('MGH')]
    site_03 = FinalData[FinalData['site'].str.match('UMN')]
    site_04 = FinalData[FinalData['site'].str.match('UNM')]

    # send the total number of sites information to each site (Remote)
    unique_sites = FinalData['site'].unique()
    unique_sites.sort()
    site_dummy = pd.get_dummies(unique_sites, drop_first=True)
    site_dummy.set_index(unique_sites, inplace=True)
    site_dummy = site_dummy.add_prefix('site_')
    site_dummy['site'] = site_dummy.index

    site_01_X, site_01_y = select_and_drop_cols(site_dummy, site_01)
    site_02_X, site_02_y = select_and_drop_cols(site_dummy, site_02)
    site_03_X, site_03_y = select_and_drop_cols(site_dummy, site_03)
    site_04_X, site_04_y = select_and_drop_cols(site_dummy, site_04)

    site_01_y1 = site_01_y.as_matrix(columns=None)
    site_02_y1 = site_02_y.as_matrix(columns=None)
    site_03_y1 = site_03_y.as_matrix(columns=None)
    site_04_y1 = site_04_y.as_matrix(columns=None)

    X1 = get_dummies_and_augment(site_01_X)
    X2 = get_dummies_and_augment(site_02_X)
    X3 = get_dummies_and_augment(site_03_X)
    X4 = get_dummies_and_augment(site_04_X)

    column_name_list = X1.columns.tolist()

    X1 = X1.values
    X2 = X2.values
    X3 = X3.values
    X4 = X4.values

    return (X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1,
            column_name_list)
