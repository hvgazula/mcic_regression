# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import shelve
central = shelve.open('multishot_results_perTol')
params_cen = central['params']
pvalues_cen = central['pvalues']
tvalues_cen = central['tvalues']
rsquared_cen = central['rsquared']
central.close()

results = shelve.open('decentralized_results')
params_decen1 = results['params']
pvalues_decen1 = results['pvalues']
tvalues_decen1 = results['tvalues']
rsquared_decen1 = results['rsquared']

site1_params_decen1 = results['site1_params']
site1_pvalues_decen1 = results['site1_pvalues']
site1_tvalues_decen1 = results['site1_tvalues']
site1_rsquared_decen1 = results['site1_rsquared']

site2_params_decen1 = results['site2_params']
site2_pvalues_decen1 = results['site2_pvalues']
site2_tvalues_decen1 = results['site2_tvalues']
site2_rsquared_decen1 = results['site2_rsquared']

site3_params_decen1 = results['site3_params']
site3_pvalues_decen1 = results['site3_pvalues']
site3_tvalues_decen1 = results['site3_tvalues']
site3_rsquared_decen1 = results['site3_rsquared']

site4_params_decen1 = results['site4_params']
site4_pvalues_decen1 = results['site4_pvalues']
site4_tvalues_decen1 = results['site4_tvalues']
site4_rsquared_decen1 = results['site4_rsquared']
results.close()

import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()
plt.plot(params_cen['diagnosis_Patient'], params_decen1['diagnosis_Patient'],
         '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Harsh')
plt.title('diagnosis_Patient_params')

plt.figure()
plt.plot(params_cen['diagnosis_Patient'], params_decen2['diagnosis_Patient'],
         '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('diagnosis_Patient_params')

plt.figure()
plt.plot(params_cen['age'], params_decen1['age'], '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Harsh')
plt.title('Age_params')

plt.figure()
plt.plot(params_cen['age'], params_decen2['age'], '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('diagnosis_Patient_params')

plt.figure()
plt.plot(pvalues_cen['diagnosis_Patient'], pvalues_decen1['diagnosis_Patient'],
         '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Harsh')
plt.title('diagnosis_Patient_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_cen['diagnosis_Patient']),
         -np.log10(pvalues_decen1['diagnosis_Patient']), '.')
plt.xlabel(r'\textit{p}-values (Centralized R')
plt.ylabel('Decentralized_Harsh')
plt.title('diagnosis_Patient_nplog10_pvalues')

plt.figure()
plt.plot(pvalues_cen['diagnosis_Patient'], pvalues_decen2['diagnosis_Patient'],
         '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('diagnosis_Patient_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_cen['diagnosis_Patient']),
         -np.log10(pvalues_decen2['diagnosis_Patient']), '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('diagnosis_Patient_nplog10_pvalues')

plt.figure()
plt.plot(pvalues_cen['age'], pvalues_decen1['age'], '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Harsh')
plt.title('age_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_cen['age']), -np.log10(pvalues_decen1['age']), '.')
plt.xlabel(r'Centralized Regression')
plt.ylabel(r'Decentralized Regression')
plt.title(r'Plot of $-\log_{10}{(\textit{p})}$ for Age')

plt.figure()
plt.plot(pvalues_cen['age'], pvalues_decen2['age'], '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('age_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_cen['age']), -np.log10(pvalues_decen2['age']), '.')
plt.xlabel('Centralized')
plt.ylabel('Decentralized_Actual')
plt.title('age_nplog10_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_decen1['age']), -np.log10(pvalues_decen2['age']),
         '.')
plt.xlabel('Decentralized_Harsh')
plt.ylabel('Decentralized_Actual')
plt.title('diagnosis_Patient_nplog10_pvalues')

plt.figure()
plt.plot(-np.log10(pvalues_decen1['diagnosis_Patient']),
         -np.log10(pvalues_decen2['diagnosis_Patient']), '.')
plt.xlabel('Decentralized_Harsh')
plt.ylabel('Decentralized_Actual')
plt.title('age_nplog10_pvalues')

# new plots from 01142018
plt.figure()
plt.plot(-np.log10(pvalues_cen['age']), -np.log10(pvalues_decen1['age']), '.')
plt.xlabel(r'Centralized Regression')
plt.ylabel(r'Decentralized Regression (Weighted Average)')
plt.title(r'Plot of $-\log_{10}{(\textit{p})}$ for Age')

plt.figure()
plt.plot(-np.log10(pvalues_cen['diagnosis_Patient']),
         -np.log10(pvalues_decen1['diagnosis_Patient']), '.')
plt.xlabel(r'Centralized Regression')
plt.ylabel(r'Decentralized Regression (Weighted Average)')
plt.title(r'Plot of $-\log_{10}{(\textit{p})}$ for Diagnosis (Control = 0)')

plt.figure()
plt.plot(-np.log10(pvalues_cen['sex_M']), -np.log10(pvalues_decen1['sex_M']),
         '.')
plt.xlabel(r'Centralized Regression')
plt.ylabel(r'Decentralized Regression (Weighted Average)')
plt.title(r'Plot of $-\log_{10}{(\textit{p})}$ for Gender (Female = 0)')