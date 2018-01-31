#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:52:40 2018

@author: Harshvardhan Gazula
@notes: Contains multi-shot regression with vanilla gradient descent
        # modified the code to restart the gradient descent if the learning rate is too high
"""

import os
import pickle
import shelve
from progressbar import ProgressBar
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm

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
    """Add a constant column and get dummies for categorical values"""
    X = pd.get_dummies(site_X, drop_first='True')
    X = sm.add_constant(X, has_constant='add')
    return X


folder_index = input('Enter the Folder name: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open("final_data_resampled.pkl", "rb") as f:
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

X1 = get_dummies_and_augment(site_01_X)
X2 = get_dummies_and_augment(site_02_X)
X3 = get_dummies_and_augment(site_03_X)
X4 = get_dummies_and_augment(site_04_X)


def gottol(vector, tol=1e-5):
    """Check if the gradient meets the tolerances"""
    return np.sum(np.square(vector)) <= tol


def objective(weights, X, y, lamb=0.0):
    """calculates the Objective function value"""
    return (1 / 2 * len(X)) * np.sum(
        (np.dot(X, weights) - y)**2) + lamb * np.linalg.norm(
            weights, ord=2) / 2.


def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
    return (1 / len(X)) * np.dot(X.T, np.dot(X, weights) - y) + lamb * weights


params, pvalues, tvalues, rsquared = [], [], [], []

fh = open(os.path.join(folder_name, 'new_vanilla_output'), 'w')
fh.write('{:^7s} {:^15s} {:^20s} {:^20s} {:^15s} {:^15s} {:^4s} \n'.format(
    'voxel', 'iterations', 'prev_obj_val', 'curr_obj_val', 'eta', 'grad_norm',
    'flag'))
fh.close()

for voxel in pbar(voxels.columns):
    flag = 0
    y1 = site_01_y[voxel]
    y2 = site_02_y[voxel]
    y3 = site_03_y[voxel]
    y4 = site_04_y[voxel]

    # Initialize at remote
    wp = np.zeros(X1.shape[1])
    prev_obj_remote = np.inf
    grad_remote = np.random.rand(X1.shape[1])
    tol = 1e-3  # 0.5e-3
    eta = 5e-3

    count = 0
    while not gottol(grad_remote, tol):
        count = count + 1

        # At local
        grad_local1 = gradient(wp, X1, y1, lamb=0)
        grad_local2 = gradient(wp, X2, y2, lamb=0)
        grad_local3 = gradient(wp, X3, y3, lamb=0)
        grad_local4 = gradient(wp, X4, y4, lamb=0)

        obj_local1 = objective(wp, X1, y1, lamb=0)
        obj_local2 = objective(wp, X2, y2, lamb=0)
        obj_local3 = objective(wp, X3, y3, lamb=0)
        obj_local4 = objective(wp, X4, y4, lamb=0)

        # at remote
        curr_obj_remote = obj_local1 + obj_local2 + obj_local3 + obj_local4
        grad_remote = (
            grad_local1 + grad_local2 + grad_local3 + grad_local4) / 4

        wc = wp - eta * grad_remote

#        print('{:07d} {:^15d} {:^20.6f} {:^20.6f} {:^15.5f} {:^15.7f} {:^4d}'.
#          format(voxel, count, prev_obj_remote, curr_obj_remote, eta,
#                 np.sum(np.square(grad_remote)), flag))

        if curr_obj_remote > prev_obj_remote:  # 11
            eta = np.around(eta - eta * (25 / 100), decimals=4)  # 12
            # start from scratch
            wp = np.zeros(X1.shape[1])
            prev_obj_remote = np.inf
            grad_remote = np.random.rand(X1.shape[1])
            if eta < 10e-10:
                break
            continue
        else:  # 13
            prev_prev = prev_obj_remote
            prev_obj_remote = curr_obj_remote

            # 9
            wp = wc

    if curr_obj_remote != prev_obj_remote or np.sum(
            np.square(grad_remote)) > tol or eta != 0.5e-3:
        flag = 1

    with open(os.path.join(folder_name, 'new_vanilla_output'), 'a') as fh:
        fh.write(
            '{:07d} {:^15d} {:^20.6f} {:^20.6f} {:^15.7f} {:^15.5f} {:^4d} \n'.
            format(voxel, count, prev_prev, curr_obj_remote, eta,
                   np.sum(np.square(grad_remote)), flag))

    avg_beta_vector = wc
    params.append(avg_beta_vector)

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
    count_y_local = [len(y1), len(y2), len(y3), len(y4)]

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
    SST_global = sst1 + sst3 + sst3 + sst4
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

params = pd.DataFrame(params, columns=X1.columns.tolist())
pvalues = pd.DataFrame(pvalues, columns=X1.columns.tolist())
tvalues = pd.DataFrame(tvalues, columns=X1.columns.tolist())
rsquared = pd.DataFrame(rsquared, columns=['rsquared_adj'])

# %% Write to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'vanilla_test'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
