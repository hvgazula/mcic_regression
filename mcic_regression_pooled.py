#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 3 15:18:33 2018

@author: Harshvardhan Gazula
@notes: Modified to remove sites information and perform pooled regression
"""

import os
import pickle
import shelve
from progressbar import ProgressBar
import pandas as pd
import statsmodels.api as sm

pbar = ProgressBar()

folder_index = input(
    'Enter the Folder name where you want your results to be saved: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open("final_data_resampled.pkl", "rb") as f:
    demographics, voxels = pickle.load(f)

demographics.drop('site', axis=1, inplace=True)

# %% Statistical Analysis
X = pd.get_dummies(demographics, drop_first=True)  # Creating Dummies
X = sm.add_constant(X)  # Augmenting the Design matrix

params, pvalues, tvalues, rsquared = [], [], [], []
for voxel in pbar(voxels.columns):
    y = voxels[voxel]

    model = sm.OLS(y, X.astype(float)).fit()
    params.append(model.params)
    pvalues.append(model.pvalues)
    tvalues.append(model.tvalues)
    rsquared.append(model.rsquared_adj)

params = pd.concat(params, axis=1).T
pvalues = pd.concat(pvalues, axis=1).T
tvalues = pd.concat(tvalues, axis=1).T
rsquared = pd.DataFrame(rsquared, columns=['rsquared_adj'])

# %% Writing to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'centralized_results'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
