#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:32:34 2018

@author: hgazula
"""

import pickle
import pandas as pd
import statsmodels.api as sm


def select_and_drop_cols(site_dummy, site_data, flag):
    """Select and crop columns"""
    if flag == 'multishot':
        select_column_list = [
            'age', 'site_MGH', 'site_UMN', 'site_UNM', 'diagnosis', 'sex'
        ]
        site_data = site_dummy.merge(site_data, on='site', how='right')
    elif flag == 'singleshot':
        select_column_list = ['age', 'diagnosis', 'sex']
    else:
        print('so such algorithm exists')

    site_data = site_data.drop('site', axis=1)
    site_X = site_data[select_column_list]
    site_y = site_data.drop(select_column_list, axis=1)
    return site_X, site_y


def get_dummies_and_augment(site_X):
    """Add a constant column and get dummies for categorical values"""
    X = pd.get_dummies(site_X, drop_first='True')
    X = sm.add_constant(X, has_constant='add')
    return X


def some_manipulation(data, site, site_dummy, algo):
    """I really dont kniw what' happening here"""
    site_data = data[data['site'].str.match(site)]
    site_X, site_y = select_and_drop_cols(site_dummy, site_data, algo)
    site_y1 = site_y.values
    X1 = get_dummies_and_augment(site_X)
    column_list = X1.columns.tolist()
    X1 = X1.values
    site_y1 = site_y1.astype('float64')

    return X1, site_y1, column_list


def gather_dummy_info(data):
    """same here don't know"""
    # send the total number of sites information to each site (Remote)
    unique_sites = data['site'].unique()
    unique_sites.sort()
    dummy = pd.get_dummies(unique_sites, drop_first=True)
    dummy.set_index(unique_sites, inplace=True)
    dummy = dummy.add_prefix('site_')
    dummy['site'] = dummy.index

    return dummy


def load_data():
    """Loads the pickle file and prepares it for further processing
    """
    with open("final_data.pkl", "rb") as file_h:
        demographics, voxels = pickle.load(file_h)

    final_data = pd.concat([demographics, voxels], axis=1)

    site_dummy = gather_dummy_info(final_data)

    shot = 'singleshot'
    X1, site_01_y1, column_name_list = some_manipulation(
        final_data, 'IA', site_dummy, shot)
    X2, site_02_y1, _ = some_manipulation(final_data, 'MGH', site_dummy, shot)
    X3, site_03_y1, _ = some_manipulation(final_data, 'UMN', site_dummy, shot)
    X4, site_04_y1, _ = some_manipulation(final_data, 'UNM', site_dummy, shot)

    return (X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1,
            column_name_list)
