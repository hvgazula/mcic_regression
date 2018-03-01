#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:52:40 2018

@author: Harshvardhan Gazula
@notes: Contains multi-shot regression with vanilla gradient descent
        # modified the code to restart the gradient descent if the learning 
        rate is too high
        # numba code for multishot learning
        # cuda code for vanilla GD
"""

import os
import pickle
import shelve
from numba import jit, vectorize
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm


def select_and_drop_cols(site_dummy, site_data):
    """Select and crop columns"""
    select_column_list = ['age', 'diagnosis', 'sex']
    #    select_column_list = [
    #        'age', 'site_MGH', 'site_UMN', 'site_UNM', 'diagnosis', 'sex'
    #    ]    
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


@jit(nopython=True)
def gottol(vector, tol=1e-5):
    """Check if the gradient meets the tolerances"""
    return np.sum(np.square(vector)) <= tol


@jit(nopython=True)
def objective(weights, X, y, lamb=0.0):
    """calculates the Objective function value"""
    return (1 / 2 * len(X)) * np.sum(
        (np.dot(X, weights) - y)**2) + lamb * np.linalg.norm(weights) / 2.


@jit(nopython=True)
def gradient(weights, X, y, lamb=0.0):
    """Computes the gradient"""
    return (1 / len(X)) * np.dot(X.T, np.dot(X, weights) - y) + lamb * weights


@vectorize(['float32(float32, float32, float32, float32, float32, float32, float32, float32)'], target='cuda')
def multishot_numba(site_01_y1, site_02_y1, site_03_y1, site_04_y1):
    size_y = site_01_y1.shape[1]

#    params = np.zeros((X1.shape[1], size_y))
#    tvalues = np.zeros((X1.shape[1], size_y))
#    rsquared = np.zeros(size_y)
    
    y1 = site_01_y1[:, voxel]
    y2 = site_02_y1[:, voxel]
    y3 = site_03_y1[:, voxel]
    y4 = site_04_y1[:, voxel]

    # Initialize at remote
    wp = np.zeros(X1.shape[1])
    prev_obj_remote = np.inf
    grad_remote = np.random.rand(X1.shape[1])
    tol = 1e-4  # 0.5e-3
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
        grad_remote = grad_local1 + grad_local2 + grad_local3 + grad_local4

        wc = wp - eta * grad_remote

        if curr_obj_remote > prev_obj_remote:  # 11
            #                eta = round(eta - eta * (25 / 100), 4)  # 12
            eta = eta - eta * (25 / 100)  # 12
            # start from scratch
            wp = np.zeros(X1.shape[1])
            prev_obj_remote = np.inf
            grad_remote = np.random.rand(X1.shape[1])
            if eta < 10e-9:
                break
            continue
        else:  # 13
            #                prev_prev = prev_obj_remote
            prev_obj_remote = curr_obj_remote
            wp = wc  # 9

    avg_beta_vector = wc
    params[:, voxel] = avg_beta_vector

    y1_estimate = np.dot(avg_beta_vector, X1.transpose())
    y2_estimate = np.dot(avg_beta_vector, X2.transpose())
    y3_estimate = np.dot(avg_beta_vector, X3.transpose())
    y4_estimate = np.dot(avg_beta_vector, X4.transpose())

    sse1 = np.linalg.norm(y1 - y1_estimate)**2
    sse2 = np.linalg.norm(y2 - y2_estimate)**2
    sse3 = np.linalg.norm(y3 - y3_estimate)**2
    sse4 = np.linalg.norm(y4 - y4_estimate)**2

    # At Local
    mean_y_local = np.array(
        [np.mean(y1), np.mean(y2),
         np.mean(y3), np.mean(y4)])
    count_y_local = np.array([len(y1), len(y2), len(y3), len(y4)])

    # At Remote
    mean_y_global = np.sum(
        mean_y_local * count_y_local) / np.sum(count_y_local)

    # At Local
    sst1 = np.sum(np.square(y1 - mean_y_global))
    sst2 = np.sum(np.square(y2 - mean_y_global))
    sst3 = np.sum(np.square(y3 - mean_y_global))
    sst4 = np.sum(np.square(y4 - mean_y_global))

    cov1 = X1.transpose() @ X1
    cov2 = X2.transpose() @ X2
    cov3 = X3.transpose() @ X3
    cov4 = X4.transpose() @ X4

    # PART - Finding rsquared (global)
    SSE_global = sse1 + sse2 + sse3 + sse4
    SST_global = sst1 + sst2 + sst3 + sst4
    r_squared_global = 1 - (SSE_global / SST_global)
    rsquared[voxel] = r_squared_global

    # PART - Finding p-value at the Remote
    varX_matrix_global = cov1 + cov2 + cov3 + cov4
    dof_global = np.sum(count_y_local) - len(avg_beta_vector)
    MSE = SSE_global / dof_global
    var_covar_beta_global = MSE * np.linalg.inv(varX_matrix_global)
    se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
    ts_global = avg_beta_vector / se_beta_global

    tvalues[:, voxel] = ts_global
    return (params, tvalues, rsquared, dof_global)


(params, tvalues, rsquared, dof_global) = multishot_numba(
    X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1)

ps_global = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)
pvalues = pd.DataFrame(ps_global.transpose(), columns=column_name_list)
params = pd.DataFrame(params.transpose(), columns=column_name_list)
tvalues = pd.DataFrame(tvalues.transpose(), columns=column_name_list)
rsquared = pd.DataFrame(rsquared.transpose(), columns=['rsquared_adj'])

# %% Write to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'multishot_results'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
