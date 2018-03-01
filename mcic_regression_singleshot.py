#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:12:49 2017

@author: Harshvardhan Gazula
@notes : Contains a variant of the single-shot regression proposed by Eswar.
         Drop site specific columns at each site and use global_beta_vector
         to calculate SSE at each local site.
@modified: 01/14/2018 weighted average single shot regression
"""

import os
import pickle
import shelve
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from progressbar import ProgressBar

pbar = ProgressBar()


def t_to_p(ts_beta, dof):
    """Returns the p-value for each t-statistic of the coefficient vector

    Args:
        dof (int)       : Degrees of Freedom
                          Given by len(y) - len(beta_vector)
        ts_beta (float) : t-statistic of shape [n_features +  1]

    Returns:
        p_values (float): of shape [n_features + 1]

    Comments:
        t to p value transformation(two tail)

    """
    return [2 * sp.stats.t.sf(np.abs(t), dof) for t in ts_beta]


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
    """Add a constant column and dummy columns for categorical values"""
    X = pd.get_dummies(site_X, drop_first='True')
    X = sm.add_constant(X, has_constant='add')
    return X


folder_index = input('Enter the name of the folder to save results: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open("final_data.pkl", "rb") as f:
    demographics, voxels = pickle.load(f)

FinalData = pd.concat([demographics, voxels], axis=1)

site_01 = FinalData[FinalData['site'].str.match('IA')]
site_02 = FinalData[FinalData['site'].str.match('MGH')]
site_03 = FinalData[FinalData['site'].str.match('UMN')]
site_04 = FinalData[FinalData['site'].str.match('UNM')]

# Send the total number of sites information to each site (Remote)
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

X1 = get_dummies_and_augment(site_01_X)
X2 = get_dummies_and_augment(site_02_X)
X3 = get_dummies_and_augment(site_03_X)
X4 = get_dummies_and_augment(site_04_X)

params, pvalues, tvalues, rsquared = [], [], [], []
site1_params, site2_params, site3_params, site4_params = [], [], [], []
site1_pvalues, site2_pvalues, site3_pvalues, site4_pvalues = [], [], [], []
site1_tvalues, site2_tvalues, site3_tvalues, site4_tvalues = [], [], [], []
site1_rsquared, site2_rsquared, site3_rsquared, site4_rsquared = [], [], [], []

for voxel in pbar(voxels.columns):

    y1 = site_01_y[voxel]
    y2 = site_02_y[voxel]
    y3 = site_03_y[voxel]
    y4 = site_04_y[voxel]

    # Start of single shot regression
    # PART 01 - Regression at Each local site
    model1 = sm.OLS(y1, X1.astype(float)).fit()
    model2 = sm.OLS(y2, X2.astype(float)).fit()
    model3 = sm.OLS(y3, X3.astype(float)).fit()
    model4 = sm.OLS(y4, X4.astype(float)).fit()

    # PART 02 - Aggregating parameter values at the remote
    sum_params = [model1.params, model2.params, model3.params, model4.params]
    count_y_local = [len(y1), len(y2), len(y3), len(y4)]
    # Weighted Average
    avg_beta_vector = np.average(sum_params, weights=count_y_local, axis=0)
#    # Simple Average
#    avg_beta_vector = np.average(sum_params, axis=0)  # Simple Average
    params.append(avg_beta_vector)

    # PART 03 - SSE at each local site
    y1_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X1.as_matrix()))
    y2_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X2.as_matrix()))
    y3_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X3.as_matrix()))
    y4_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X4.as_matrix()))

    sse1 = np.linalg.norm(y1 - y1_estimate)**2
    sse2 = np.linalg.norm(y2 - y2_estimate)**2
    sse3 = np.linalg.norm(y3 - y3_estimate)**2
    sse4 = np.linalg.norm(y4 - y4_estimate)**2

    # At Local
    mean_y_local = [np.mean(y1), np.mean(y2), np.mean(y3), np.mean(y4)]

    # At Remote
    mean_y_global = np.average(mean_y_local, weights=count_y_local)

    # At Local
    sst1 = np.sum(np.square(y1 - mean_y_global))
    sst2 = np.sum(np.square(y2 - mean_y_global))
    sst3 = np.sum(np.square(y3 - mean_y_global))
    sst4 = np.sum(np.square(y4 - mean_y_global))

    cov1 = np.matmul(np.matrix.transpose(X1.as_matrix()), X1.as_matrix())
    cov2 = np.matmul(np.matrix.transpose(X2.as_matrix()), X2.as_matrix())
    cov3 = np.matmul(np.matrix.transpose(X3.as_matrix()), X3.as_matrix())
    cov4 = np.matmul(np.matrix.transpose(X4.as_matrix()), X4.as_matrix())

    # PART 05 - Finding rsquared (global)
    SSE_global = sse1 + sse2 + sse3 + sse4
    SST_global = sst1 + sst2 + sst3 + sst4
    r_squared_global = 1 - (SSE_global / SST_global)
    rsquared.append(r_squared_global)

    # PART 04 - Finding p-value at the Remote
    varX_matrix_global = cov1 + cov2 + cov3 + cov4

    dof_global = np.sum(count_y_local) - len(avg_beta_vector)

    MSE = SSE_global / dof_global
    var_covar_beta_global = MSE * sp.linalg.inv(varX_matrix_global)
    se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
    ts_global = avg_beta_vector / se_beta_global
    ps_global = t_to_p(ts_global, dof_global)

    tvalues.append(ts_global)
    pvalues.append(ps_global)

column_names = model1.params.axes[0].tolist()

params = pd.DataFrame(params, columns=column_names)
pvalues = pd.DataFrame(pvalues, columns=column_names)
tvalues = pd.DataFrame(tvalues, columns=column_names)
rsquared = pd.DataFrame(rsquared, columns=['rsquared_adj'])

# %% Writing to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'singleshot_results'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
