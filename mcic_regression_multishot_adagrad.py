#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adagrad optimizer with no hot-start
"""

import shelve
from numba import jit, prange
import numpy as np
import scipy as sp
import os
import pandas as pd
from mcic_load_data import load_data


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
def multishot_adagrad(X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4,
                    site_04_y1):

#    size_y = site_01_y1.shape[1]
    size_y = 1

    params = np.zeros((X1.shape[1], size_y))
    tvalues = np.zeros((X1.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        y1 = site_01_y1[:, voxel]
        y2 = site_02_y1[:, voxel]
        y3 = site_03_y1[:, voxel]
        y4 = site_04_y1[:, voxel]

        # Initialize at remote
        gti = grad_remote = wp = np.zeros(X1.shape[1])

        eps = 1e-8
        tol = 1e-6
        eta = 1e-4

        count = 0
        while True:
            count = count + 1

            # At local
            grad_local1 = gradient(wp, X1, y1, lamb=0)
            grad_local2 = gradient(wp, X2, y2, lamb=0)
            grad_local3 = gradient(wp, X3, y3, lamb=0)
            grad_local4 = gradient(wp, X4, y4, lamb=0)

            # at remote
            grad_remote = grad_local1 + grad_local2 + grad_local3 + grad_local4

            gti += np.square(grad_remote)
            adjusted_grad_denom = (eps + np.sqrt(gti))
            grad_remote = grad_remote/adjusted_grad_denom

            wc = wp - eta * grad_remote

            print(np.linalg.norm(wc - wp))
            if np.linalg.norm(wc - wp) <= tol:
                break

            wp = wc

        print(voxel, count)

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


folder_index = input('Enter the name of the folder to save results: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1, column_name_list = load_data(
)

(params, tvalues, rsquared, dof_global) = multishot_adagrad(
    X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1)

ps_global = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)
pvalues = pd.DataFrame(ps_global.transpose(), columns=column_name_list)
params = pd.DataFrame(params.transpose(), columns=column_name_list)
tvalues = pd.DataFrame(tvalues.transpose(), columns=column_name_list)
rsquared = pd.DataFrame(rsquared.transpose(), columns=['rsquared_adj'])

# %% Write to a file
print('Writing data to a shelve file')
results = shelve.open(
    os.path.join(folder_name, 'multishot_results_adagrad'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
