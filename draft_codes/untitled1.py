# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nibabel as nib
import numpy as np
import os
import shelve
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

folder_name = 'Results'

pooled = shelve.open(os.path.join(folder_name, 'centralizednumba'))
params_pooled = pooled['params']
pvalues_pooled = pooled['pvalues']
tvalues_pooled = pooled['tvalues']
rsquared_pooled = pooled['rsquared']
sse_pooled = pooled['sse']
pooled.close()

singleshot = shelve.open(os.path.join(folder_name, 'singleshotWA'))
params_singleshot = singleshot['params']
pvalues_singleshot = singleshot['pvalues']
tvalues_singleshot = singleshot['tvalues']
rsquared_singleshot = singleshot['rsquared']
sse_singleshot = singleshot['sse']
singleshot.close()

multishot = shelve.open(os.path.join(folder_name, 'multishotsite'))
params_multishot = multishot['params']
pvalues_multishot = multishot['pvalues']
tvalues_multishot = multishot['tvalues']
rsquared_multishot = multishot['rsquared']
sse_multishot = multishot['sse']
multishot.close()

multishotexact = shelve.open(os.path.join(folder_name, 'multishotExact'))
params_multishotexact = multishotexact['params']
pvalues_multishotexact = multishotexact['pvalues']
tvalues_multishotexact = multishotexact['tvalues']
rsquared_multishotexact = multishotexact['rsquared']
sse_multishotexact = multishotexact['sse']
multishotexact.close()
# %% Head Directory where the data is stored and loading the mask
data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask')
mask = nib.load(os.path.join(mask_location, 'mask.nii'))

# =============================================================================
# PAIRPLOT for pairwise SSE
# =============================================================================
sns.set(style="ticks")
abc = pd.DataFrame()
abc["pooled"] = sse_pooled["sse"]
abc["singleshot"] = sse_singleshot["sse"]
abc["multishot"] = sse_multishot["sse"]
sns.pairplot(abc)
# =============================================================================
# PAIRPLOT for rsquared
# =============================================================================
sns.set(style="ticks")
abcr2 = pd.DataFrame()
abcr2["pooled"] = rsquared_pooled["rsquared_adj"]
abcr2["singleshot"] = rsquared_singleshot["rsquared_adj"]
abcr2["multishot"] = rsquared_multishot["rsquared_adj"]
sns.pairplot(abcr2)
# =============================================================================
# PAIRPLOT for const
# =============================================================================
sns.set(style="ticks")
abccon = pd.DataFrame()
abccon["pooled"] = params_pooled["const"]
abccon["singleshot"] = params_singleshot["const"]
abccon["multishot"] = params_multishot["const"]
lm4 = sns.pairplot(abccon, diag_kind="kde")
# =============================================================================
# PAIRPLOT for beta_age
# =============================================================================
sns.set(style="ticks")
abcba = pd.DataFrame()
abcba["pooled"] = params_pooled["age"]
abcba["singleshot"] = params_singleshot["age"]
abcba["multishot"] = params_multishot["age"]
lm1 = sns.pairplot(abcba, diag_kind="kde")
# =============================================================================
# PAIRPLOT for beta_diagnosis
# =============================================================================
sns.set(style="ticks")
abcbd = pd.DataFrame()
abcbd["pooled"] = params_pooled["diagnosis_Patient"]
abcbd["singleshot"] = params_singleshot["diagnosis_Patient"]
abcbd["multishot"] = params_multishot["diagnosis_Patient"]
lm2 = sns.pairplot(abcbd)
# =============================================================================
# PAIRPLOT for beta_sex
# =============================================================================
sns.set(style="ticks")
abcbs = pd.DataFrame()
abcbs["pooled"] = params_pooled["sex_M"]
abcbs["singleshot"] = params_singleshot["sex_M"]
abcbs["multishot"] = params_multishot["sex_M"]
lm3 = sns.pairplot(abcbs)
# =============================================================================
# # PAIRPLOT for site_MGH beta
# =============================================================================
sns.set(style="ticks")
abcmgh = pd.DataFrame()
abcmgh["pooled"] = params_pooled["site_MGH"]
abcmgh["multishot"] = params_multishot["site_MGH"]
lm5 = sns.pairplot(abcmgh, diag_kind="kde")
# =============================================================================
# # PAIRPLOT for site_UMN beta
# =============================================================================
sns.set(style="ticks")
abcumn = pd.DataFrame()
abcumn["pooled"] = params_pooled["site_UMN"]
abcumn["multishot"] = params_multishot["site_UMN"]
lm6 = sns.pairplot(abcumn, diag_kind="kde")
# =============================================================================
# # PAIRPLOT for site_UNM beta
# =============================================================================
sns.set(style="ticks")
abcunm = pd.DataFrame()
abcunm["pooled"] = params_pooled["site_UNM"]
abcunm["multishot"] = params_multishot["site_UNM"]
lm7 = sns.pairplot(abcunm, diag_kind="kde")
# =============================================================================
# VIOLIN PLOTS for SSE
# =============================================================================
sse_pooled['label'] = 'pooled'
sse_singleshot['label'] = 'singleshot'
sse_multishot['label'] = 'multishot'

sse_new = pd.concat([sse_pooled, sse_singleshot, sse_multishot])

ax1 = sns.violinplot(x="label", y="sse", data=sse_new)
ax1.set_xlabel("Type of Regression")
ax1.set_ylabel("Sum Square of Errors")
ax1.set_title('Violin plot of Sum Square of Errors for each Regression')
# =============================================================================
# VIOLIN PLOTS for sse differences
# =============================================================================
sse1 = pd.DataFrame()
sse2 = pd.DataFrame()
sse3 = pd.DataFrame()
sse1['diff'] = sse_pooled['sse'] - sse_singleshot['sse']
sse2['diff'] = sse_pooled['sse'] - sse_multishot['sse']
sse3['diff'] = sse_singleshot['sse'] - sse_multishot['sse']
sse1['label'] = 'P - SS'
sse2['label'] = 'P - MS'
sse3['label'] = 'SS - MS'

sse_new2 = pd.concat([sse1, sse2, sse3])

sns.set_style('darkgrid')
ax2 = sns.violinplot(x="label", y="diff", data=sse_new2)
ax2.set_xlabel("Regression pairs (P - pooled, SS - Singleshot, MS - Multishot)")
ax2.set_ylabel("Diff. of Sum Square of Errors")
ax2.set_title('Violin plot of SSE differences for every pair of regression')
# =============================================================================
# plotting violin plots for sse comparisons
# =============================================================================
mat1 = pd.concat([sse_pooled, sse_singleshot])
mat1['diff_label'] = 'P_vs_SS'

mat2 = pd.concat([sse_pooled, sse_multishot])
mat2['diff_label'] = 'P_vs_MS'

mat3 = pd.concat([sse_singleshot, sse_multishot])
mat3['diff_label'] = 'SS_vs_MS'

mat = pd.concat([mat1, mat2, mat3])
ax3 = sns.violinplot(x="diff_label", y="sse", hue="label", data=mat)
ax3.set_xlabel("Pairs of Regression")
ax3.set_ylabel('Sum Square of Errors')
ax3.set_title('Comparison of SSE for every pair of regression')
ax3.legend(
    bbox_to_anchor=(0, -0.098, 1., -.102),
    loc=9,
    ncol=3,
    mode="expand",
    borderaxespad=0.)
# =============================================================================
# Violin plots of rsquared for each regression
# =============================================================================
sns.set_style("darkgrid")
rsquared_pooled['label'] = 'Pooled'
rsquared_singleshot['label'] = 'Single-shot'
rsquared_multishot['label'] = 'Multi-shot'

rsquared_new = pd.concat(
    [rsquared_pooled, rsquared_singleshot, rsquared_multishot])
ax1 = sns.violinplot(x="label", y="rsquared_adj", data=rsquared_new)
ax1.set_xlabel("Type of Regression")
ax1.set_ylabel("Adjusted R-square")
ax1.set_title('Violin plot of Adjusted R-squared for each Regression')
# =============================================================================
# pvalues
# =============================================================================
pvalues_pooled1 = -np.log10(pvalues_pooled)
pvalues_singleshot1 = -np.log10(pvalues_singleshot)
pvalues_multishot1 = -np.log10(pvalues_multishot)

pvalues_pooled1['label'] = 'Pooled'
pvalues_singleshot1['label'] = 'Single-shot'
pvalues_multishot1['label'] = 'Multi-shot'

mat_old = pd.concat([pvalues_pooled1, pvalues_singleshot1, pvalues_multishot])
plt.figure()
ax8 = sns.violinplot(x="label", y="age", data=mat)
ax8.set_xlabel("Type of Regression")
ax8.set_ylabel("-$\log_{10} p$-value for Age")
ax8.set_title(
    "Violin plot of -$\log_{10} p$-values from each type of regression")

mat1 = pd.concat([pvalues_pooled1, pvalues_singleshot1])
mat1['diff_label'] = 'P_vs_SS'

mat2 = pd.concat([pvalues_pooled1, pvalues_multishot1])
mat2['diff_label'] = 'P_vs_MS'

mat3 = pd.concat([pvalues_singleshot1, pvalues_multishot1])
mat3['diff_label'] = 'SS_vs_MS'

mat = pd.concat([mat2, mat1, mat3])

plt.figure()
ax5 = sns.violinplot(x="diff_label", y="age", hue="label", data=mat)
ax5.set_xlabel("Pair of Regressions")
ax5.set_ylabel("Age")
ax5.set_title(
    "Comparison of -$\log_{10} p$-values for every pair of regression")
ax5.legend(
    bbox_to_anchor=(0., -0.098, 1., -.102),
    loc=9,
    ncol=3,
    mode="expand",
    borderaxespad=0.)

plt.figure()
ax6 = sns.violinplot(
    x="diff_label", y="diagnosis_Patient", hue="label", data=mat)
ax6.set_xlabel("Pair of Regressions")
ax6.set_ylabel("Diagnosis")
ax6.set_title(
    "Comparison of -$\log_{10} p$-values for every pair of regression")
ax6.legend(
    bbox_to_anchor=(0., -0.098, 1., -.102),
    loc=9,
    ncol=3,
    mode="expand",
    borderaxespad=0.)

plt.figure()
ax7 = sns.violinplot(x="diff_label", y="sex_M", hue="label", data=mat)
ax7.set_xlabel("Pair of Regressions")
ax7.set_ylabel("Gender")
ax7.set_title(
    "Comparison of -$\log_{10} p$-values for every pair of regression")
ax7.legend(
    bbox_to_anchor=(0., -0.098, 1., -.102),
    loc=9,
    ncol=3,
    mode="expand",
    borderaxespad=0.)

# =============================================================================
# BRAIN IMAGES
# =============================================================================

# =============================================================================
# Plotting SSE
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_pooled['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized_sse_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_singleshot['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot_sse_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_multishot['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'multishot_sse_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and singleshot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_pooled['sse'] - sse_singleshot['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized-singleshot'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_pooled['sse'] - sse_multishot['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized-multishot'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between singleshot and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = sse_singleshot['sse'] - sse_multishot['sse']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot-multishot'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Plotting adjusted-rsquared
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = rsquared_pooled['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized_rsquared_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = rsquared_singleshot['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot_rsquared_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = rsquared_multishot['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'multishot_rsquared_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and singleshot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = rsquared_pooled['rsquared_adj'] - rsquared_singleshot['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized-singleshot_rsquared'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = rsquared_pooled['rsquared_adj'] - rsquared_multishot['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized-multishot_rsquared'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between singleshot and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = rsquared_singleshot['rsquared_adj'] - rsquared_multishot['rsquared_adj']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot-multishot_rsquared'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Plotting params['Age']
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_pooled['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized_age_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_singleshot['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot_age_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_multishot['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'multishot_age_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and singleshot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_pooled['age'] - params_singleshot['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-singleshot_age_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_pooled['age'] - params_multishot['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-multishot_age_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between singleshot and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() >
         0] = params_singleshot['age'] - params_multishot['age']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot-multishot_age_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Plotting params['diagnosis_Patient']
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_pooled['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized_diagnosis_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_singleshot['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot_diagnosis_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_multishot['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'multishot_diagnosis_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and singleshot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = params_pooled['diagnosis_Patient'] - params_singleshot['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-singleshot_diagnosis_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = params_pooled['diagnosis_Patient'] - params_multishot['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-multishot_diagnosis_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between singleshot and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[
    mask.get_data() >
    0] = params_singleshot['diagnosis_Patient'] - params_multishot['diagnosis_Patient']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot-multishot_diagnosis_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Plotting params['sex_M']
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_pooled['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'centralized_gender_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_singleshot['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot_gender_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

new_data = np.zeros(mask.shape)
new_data[mask.get_data() > 0] = params_multishot['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'multishot_gender_beta_on_brain'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and singleshot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() >
         0] = params_pooled['sex_M'] - params_singleshot['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-singleshot_gender_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between pooled and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() >
         0] = params_pooled['sex_M'] - params_multishot['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'pooled-multishot_gender_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)

# =============================================================================
# Difference in sse between singleshot and multishot
# =============================================================================
new_data = np.zeros(mask.shape)
new_data[mask.get_data() >
         0] = params_singleshot['sex_M'] - params_multishot['sex_M']
clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
image_string = 'singleshot-multishot_gender_beta'
print('Saving ', image_string)
nib.save(clipped_img, image_string)
