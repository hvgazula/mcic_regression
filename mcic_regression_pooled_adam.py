#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 3 15:18:33 2018

@author: Harshvardhan Gazula
@notes: Modified to remove sites information and perform pooled regression
"""
import numpy as np
import os
import pickle
import pandas as pd
import statsmodels.api as sm
import shelve
from numba import jit, prange
import scipy as sp


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


@jit(nopython=True)
def pooled_adam(X, y):
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)
    
    for voxel in range(size_y):
        print(voxel)
        curr_y = y[:, voxel]
        
        # Initialize at remote
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        wp = np.zeros(X1.shape[1])
        mt = np.zeros(X1.shape[1])
        vt = np.zeros(X1.shape[1])

        tol = 1e-8
        eta = 1e-2

        count = 0
        while True:
            count = count + 1

            grad_remote = gradient(wp, X1, curr_y, lamb=0)
            mt = beta1 * mt + (1 - beta1) * grad_remote
            vt = beta2 * vt + (1 - beta2) * (grad_remote**2)

            m = mt / (1 - beta1**count)
            v = vt / (1 - beta2**count)

            wc = wp - eta * m / (np.sqrt(v) + eps)

            if np.linalg.norm(wc - wp) <= tol:
                break

            wp = wc
    
        beta_vector = wc
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate)**2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(X.T @ X)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


folder_index = input(
    'Enter the Folder name where you want your results to be saved: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

with open("final_data.pkl", "rb") as f:
    demographics, voxels = pickle.load(f)

# =============================================================================
## To Drop site info from pooled data
# demographics.drop('site', axis=1, inplace=True)
# =============================================================================

# %% Statistical Analysis
X = pd.get_dummies(demographics, drop_first=True)  # Creating Dummies
X = sm.add_constant(X)  # Augmenting the Design matrix
column_name_list = X.columns.tolist()

X1 = X.values
y1 = voxels.values.astype('float64')

(params, sse, tvalues, rsquared, dof_global) = pooled_adam(X1, y1)

ps_global = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)
pvalues = pd.DataFrame(ps_global.transpose(), columns=column_name_list)
sse = pd.DataFrame(sse.transpose(), columns=['sse'])
params = pd.DataFrame(params.transpose(), columns=column_name_list)
tvalues = pd.DataFrame(tvalues.transpose(), columns=column_name_list)
rsquared = pd.DataFrame(rsquared.transpose(), columns=['rsquared_adj'])

# %% Writing to a file
print('Writing data to a shelve file')
results = shelve.open(os.path.join(folder_name, 'centralizedAdam'))
results['params'] = params
results['sse'] = sse
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
