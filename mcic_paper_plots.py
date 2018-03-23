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


def read_results_pkl(folder_name, filename):

    p = shelve.open(os.path.join(folder_name, filename))
    params = p['params']
    pvals = p['pvalues']
    tvals = p['tvalues']
    rsquared = p['rsquared']
    sse = p['sse']
    p.close()

    return (params, pvals, tvals, rsquared, sse)


folder_name = 'Results'

# %% Head Directory where the data is stored and loading the mask
data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask')
mask = nib.load(os.path.join(mask_location, 'mask.nii'))

(params_pooled, pvalues_pooled, tvalues_pooled, rsquared_pooled,
 sse_pooled) = read_results_pkl(folder_name, 'centralizednumba')
(params_singleshot, pvalues_singleshot, tvalues_singleshot,
 rsquared_singleshot, sse_singleshot) = read_results_pkl(
     folder_name, 'singleshotWA')
(params_multishot, pvalues_multishot, tvalues_multishot, rsquared_multishot,
 sse_multishot) = read_results_pkl(folder_name, 'multishotsite')
(params_multishotexact, pvalues_multishotexact, tvalues_multishotexact,
 rsquared_multishotexact, sse_multishotexact) = read_results_pkl(
     folder_name, 'multishotExact')

# =============================================================================
# PAIRPLOT for pairwise SSE
# =============================================================================
sns.set(style="ticks")
abc = pd.DataFrame()
abc["pooled"] = sse_pooled["sse"]
abc["singleshot"] = sse_singleshot["sse"]
abc["multishot"] = sse_multishot["sse"]
pltsse = sns.pairplot(abc)
plt.suptitle('Pair plot of SSE')
pltsse.savefig("sse.png")
#plt.close()
# =============================================================================
# PAIRPLOT for rsquared
# =============================================================================
sns.set(style="ticks")
abcr2 = pd.DataFrame()
abcr2["pooled"] = rsquared_pooled["rsquared_adj"]
abcr2["singleshot"] = rsquared_singleshot["rsquared_adj"]
abcr2["multishot"] = rsquared_multishot["rsquared_adj"]
pltr2 = sns.pairplot(abcr2)
plt.suptitle('Pair plot of R-squared')
pltr2.savefig("rsquared.png")
#plt.close()
# =============================================================================
# PAIRPLOT for const
# =============================================================================
abccon = pd.DataFrame()
abccon["pooled"] = params_pooled["const"]
abccon["singleshot"] = params_singleshot["const"]
abccon["multishot"] = params_multishot["const"]
pltcon = sns.pairplot(abccon)
plt.suptitle('Pair plot of beta_constant')
pltcon.savefig("beta_con.png")
#plt.close(pltcon)
# =============================================================================
# PAIRPLOT for beta_age
# =============================================================================
abcba = pd.DataFrame()
abcba["pooled"] = params_pooled["age"]
abcba["singleshot"] = params_singleshot["age"]
abcba["multishot"] = params_multishot["age"]
pltage = sns.pairplot(abcba)
plt.suptitle('Pair plot of beta_age')
pltage.savefig("beta_age.png")
#plt.close(pltage)
# =============================================================================
# PAIRPLOT for beta_diagnosis
# =============================================================================
abcbd = pd.DataFrame()
abcbd["pooled"] = params_pooled["diagnosis_Patient"]
abcbd["singleshot"] = params_singleshot["diagnosis_Patient"]
abcbd["multishot"] = params_multishot["diagnosis_Patient"]
pltdiag = sns.pairplot(abcbd)
plt.suptitle('Pair plot of beta_diagnosis')
pltdiag.savefig("beta_diag.png")
#plt.close(pltdiag)
# =============================================================================
# PAIRPLOT for beta_sex
# =============================================================================
#sns.set(style="ticks")
#abcbs = pd.DataFrame()
#abcbs["pooled"] = params_pooled["sex_M"]
#abcbs["singleshot"] = params_singleshot["sex_M"]
#abcbs["multishot"] = params_multishot["sex_M"]
#lm3 = sns.pairplot(abcbs)

# =============================================================================
# # PAIRPLOT for site_07 beta
# =============================================================================
sns.set(style="ticks")
abc07 = pd.DataFrame()
abc07["pooled"] = params_pooled["site_07"]
abc07["multishot"] = params_multishot["site_07"]
plt07 = sns.pairplot(abc07)
plt.suptitle('Pair plot of beta_site_07')
plt07.savefig("beta_site_07.png")
#plt.close(plt07)
# =============================================================================
# # PAIRPLOT for site_09 beta
# =============================================================================
sns.set(style="ticks")
abc09 = pd.DataFrame()
abc09["pooled"] = params_pooled["site_09"]
abc09["multishot"] = params_multishot["site_09"]
plt09 = sns.pairplot(abc09)
plt.suptitle('Pair plot of beta_site_09')
plt09.savefig("beta_site_09.png")
#plt.close(plt09)
# =============================================================================
# # PAIRPLOT for site_10 beta
# =============================================================================
sns.set(style="ticks")
abc10 = pd.DataFrame()
abc10["pooled"] = params_pooled["site_10"]
abc10["multishot"] = params_multishot["site_10"]
plt10 = sns.pairplot(abc10)
plt.suptitle('Pair plot of beta_site_10')
plt10.savefig("beta_site_10.png")
#plt.close(plt10)
# =============================================================================
# # PAIRPLOT for site_12 beta
# =============================================================================
sns.set(style="ticks")
abc12 = pd.DataFrame()
abc12["pooled"] = params_pooled["site_12"]
abc12["multishot"] = params_multishot["site_12"]
plt12 = sns.pairplot(abc12)
plt.suptitle('Pair plot of beta_site_12')
plt12.savefig("beta_site_12.png")
#plt.close(plt12)
# =============================================================================
# # PAIRPLOT for site_13 beta
# =============================================================================
abc13 = pd.DataFrame()
abc13["pooled"] = params_pooled["site_13"]
abc13["multishot"] = params_multishot["site_13"]
plt13 = sns.pairplot(abc13)
plt.suptitle('Pair plot of beta_site_13')
plt13.savefig("beta_site_13.png")
#plt.close(plt13)
# =============================================================================
# # PAIRPLOT for site_18 beta
# =============================================================================
sns.set(style="ticks")
abc18 = pd.DataFrame()
abc18["pooled"] = params_pooled["site_18"]
abc18["multishot"] = params_multishot["site_18"]
plt18 = sns.pairplot(abc18)
plt.suptitle('Pair plot of beta_site_18')
plt18.savefig("beta_site_18.png")
#plt.close(plt18)
# =============================================================================
# # PAIRPLOT for beta_age_pval
# =============================================================================
sns.set(style="ticks")
abcconp = pd.DataFrame()
abcconp["pooled"] = pvalues_pooled["const"]
abcconp["singleshot"] = pvalues_singleshot["const"]
abcconp["multishot"] = pvalues_multishot["const"]
pltconp = sns.pairplot(abcconp)
plt.show()
plt.suptitle('Pair plot of p-values for beta_Constant')
pltconp.savefig("conp.png")
#plt.close(pltconp)

abcagep = pd.DataFrame()
abcagep["pooled"] = pvalues_pooled["age"]
abcagep["singleshot"] = pvalues_singleshot["age"]
abcagep["multishot"] = pvalues_multishot["age"]
pltagep = sns.pairplot(abcagep)
plt.suptitle('Pair plot of p-values for beta_age')
pltagep.savefig("agep.png")
#plt.close(pltagep)

abcdiagp = pd.DataFrame()
abcdiagp["pooled"] = pvalues_pooled["diagnosis_Patient"]
abcdiagp["singleshot"] = pvalues_singleshot["diagnosis_Patient"]
abcdiagp["multishot"] = pvalues_multishot["diagnosis_Patient"]
pltdiagp = sns.pairplot(abcdiagp)
plt.suptitle('Pair plot of p-values for beta_diagnosis')
pltdiagp.savefig("diagp.png")
#plt.close(pltdiagp)

abc07p = pd.DataFrame()
abc07p["pooled"] = pvalues_pooled["site_07"]
abc07p["multishot"] = pvalues_multishot["site_07"]
plt07p = sns.pairplot(abc07p)
plt.suptitle('Pair plot of p-values for beta_site_07')
plt07p.savefig("site_07p.png")
#plt.close(plt07p)

abc09p = pd.DataFrame()
abc09p["pooled"] = pvalues_pooled["site_09"]
abc09p["multishot"] = pvalues_multishot["site_09"]
plt09p = sns.pairplot(abc09p)
plt.suptitle('Pair plot of p-values for beta_site_09')
plt09p.savefig("site_09p.png")
#plt.close(plt09p)

abc10p = pd.DataFrame()
abc10p["pooled"] = pvalues_pooled["site_10"]
abc10p["multishot"] = pvalues_multishot["site_10"]
plt10p = sns.pairplot(abc10p)
plt.suptitle('Pair plot of p-values for beta_site_10')
plt10p.savefig("site_10p.png")
#plt.close(plt10p)

abc12p = pd.DataFrame()
abc12p["pooled"] = pvalues_pooled["site_12"]
abc12p["multishot"] = pvalues_multishot["site_12"]
plt12p = sns.pairplot(abc12p)
plt.suptitle('Pair plot of p-values for beta_site_12')
plt12p.savefig("site_12p.png")
#plt.close(plt12p)

abc13p = pd.DataFrame()
abc13p["pooled"] = pvalues_pooled["site_13"]
abc13p["multishot"] = pvalues_multishot["site_13"]
plt13p = sns.pairplot(abc13p)
plt.suptitle('Pair plot of p-values for beta_site_13')
plt13p.savefig("site_13p.png")
#plt.close(plt13p)

abc18p = pd.DataFrame()
abc18p["pooled"] = pvalues_pooled["site_18"]
abc18p["multishot"] = pvalues_multishot["site_18"]
plt18p = sns.pairplot(abc18p)
plt.suptitle('Pair plot of p-values for beta_site_18')
plt18p.savefig("site_18p.png")
#plt.close(plt18p)
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
ax2.set_xlabel(
    "Regression pairs (P - pooled, SS - Singleshot, MS - Multishot)")
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
