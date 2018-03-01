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
import shelve
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from progressbar import ProgressBar
from mcic_load_data import load_data

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


def singleshot_regression(X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4,
                          site_04_y1):

    params, pvalues, tvalues, rsquared = [], [], [], []
    size_y = site_01_y1.shape[1]

    for voxel in pbar(range(size_y)):

        y1 = site_01_y1[:, voxel]
        y2 = site_02_y1[:, voxel]
        y3 = site_03_y1[:, voxel]
        y4 = site_04_y1[:, voxel]

        # Start of single shot regression
        # PART 01 - Regression at Each local site
        model1 = sm.OLS(y1, X1.astype(float)).fit()
        model2 = sm.OLS(y2, X2.astype(float)).fit()
        model3 = sm.OLS(y3, X3.astype(float)).fit()
        model4 = sm.OLS(y4, X4.astype(float)).fit()

        # PART 02 - Aggregating parameter values at the remote
        sum_params = [
            model1.params, model2.params, model3.params, model4.params
        ]
        count_y_local = [len(y1), len(y2), len(y3), len(y4)]

        # Weighted Average
        # avg_beta_vector = np.average(sum_params, weights=count_y_local, axis=0)
        # Simple Average
        avg_beta_vector = np.average(sum_params, axis=0)  # Simple Average

        params.append(avg_beta_vector)

        # PART 03 - SSE at each local site
        y1_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X1))
        y2_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X2))
        y3_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X3))
        y4_estimate = np.dot(avg_beta_vector, np.matrix.transpose(X4))

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

        cov1 = np.matmul(np.matrix.transpose(X1), X1)
        cov2 = np.matmul(np.matrix.transpose(X2), X2)
        cov3 = np.matmul(np.matrix.transpose(X3), X3)
        cov4 = np.matmul(np.matrix.transpose(X4), X4)

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

    return (params, pvalues, tvalues, rsquared)


folder_index = input('Enter the name of the folder to save results: ')
folder_name = folder_index.replace(' ', '_')
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1, column_name_list = load_data(
)

(params, pvalues, tvalues, rsquared) = singleshot_regression(
    X1, site_01_y1, X2, site_02_y1, X3, site_03_y1, X4, site_04_y1)

params = pd.DataFrame(params, columns=column_name_list)
pvalues = pd.DataFrame(pvalues, columns=column_name_list)
tvalues = pd.DataFrame(tvalues, columns=column_name_list)
rsquared = pd.DataFrame(rsquared, columns=['rsquared_adj'])

# %% Writing to a file
print('Writing data to a shelve file')
results = shelve.open(
    os.path.join(folder_name, 'singleshot_results_SA_resampled'))
results['params'] = params
results['pvalues'] = pvalues
results['tvalues'] = tvalues
results['rsquared'] = rsquared
results.close()
