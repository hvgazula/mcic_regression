#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 22:31:16 2018

@author: Harshvardhan Gazula
@notes: contains code to convert data back to nii images
    This code is still in development. If approved, it can be a
    replacement for the mcic_regression_print_images.m script
"""

import os
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from nistats.utils import z_score
from nistats.thresholding import map_threshold

# https://programtalk.com/vs2/python/9272/nistats/nistats/thresholding.py/
# https://15-35545854-gh.circle-artifacts.com/0/home/ubuntu/nistats/doc/_build/html/auto_examples/plot_thresholding.html


def fdr_threshold(p_vals, q):
    """ return the BH fdr for the input z_vals"""
    p = np.sort(p_vals)
    V = len(p_vals)
    i = np.arange(1, V + 1, dtype=float)
    cVID = 1
    pID = p[np.max(np.where(p <= i / V * q / cVID))]

    if pID.size:
        return pID

    return 0


def print_images1(files, masker, mask_array, curr_work_folder, coords):
    """Convert NIfTI files to images"""
    for file in files:
        pdata = nib.load(os.path.join(curr_work_folder, file)).get_data()
        pvals = pdata[mask_array > 0]
        z_map = masker.inverse_transform(z_score(np.power(10, -np.abs(pvals))))
        threshold1 = fdr_threshold(np.power(10, -np.abs(pvals)), 0.05)
        thresholded_map2, threshold2 = map_threshold(
            z_map, threshold=.05, height_control='fdr')
        print(file, threshold1, threshold2)

        # According to internet
        plotting.plot_stat_map(
            thresholded_map2,
            threshold=threshold2,
            cut_coords=coords,
            title='Thresholded z map, expected fdr = .05',
            output_file=os.path.join(curr_work_folder,
                                     os.path.splitext(file)[0] + '_01'))

        # Flop
        plotting.plot_stat_map(
            os.path.join(working_folder, file),
            threshold=threshold1,
            cut_coords=coords,
            title='Thresholded z map, expected fdr = .05',
            output_file=os.path.join(curr_work_folder,
                                     os.path.splitext(file)[0] + '_02'))

        # Does this make sense
        plotting.plot_stat_map(
            thresholded_map2,
            threshold=threshold1,
            cut_coords=coords,
            title='Thresholded z map, expected fdr = .05',
            output_file=os.path.join(curr_work_folder,
                                     os.path.splitext(file)[0] + '_03'))

        #
        plotting.plot_stat_map(
            os.path.join(working_folder, file),
            threshold=threshold2,
            cut_coords=coords,
            title='Thresholded z map, expected fdr = .05',
            output_file=os.path.join(curr_work_folder,
                                     os.path.splitext(file)[0] + '_04'))


data_location = '/export/mialab/users/hgazula/mcic_regression'
mask = os.path.join(data_location, 'mask/mask.nii')
mask_data = nib.load(mask).get_data()

folder_name = input('Please enter the output folder name: ')
working_folder = os.path.join(os.getcwd(), folder_name)

nii_files = sorted([
    i for i in os.listdir(working_folder)
    if os.path.isfile(os.path.join(working_folder, i))
    and i.startswith('pvalues') and i.endswith('.nii')
])

diagnosis_files = [i for i in nii_files if 'diagnosis' in i]
age_files = [i for i in nii_files if 'age' in i]
sex_files = [i for i in nii_files if 'sex' in i]

nifti_masker = NiftiMasker(
    smoothing_fwhm=5, memory='nilearn_cache', memory_level=1)
nifti_masker.fit_transform(mask)

print_images1(diagnosis_files, nifti_masker, mask_data, working_folder,
              (-5, 44, -12))
print_images1(age_files, nifti_masker, mask_data, working_folder,
              (-58, -18, 42))
print_images1(sex_files, nifti_masker, mask_data, working_folder, (-13, 7, 8))

#def print_images(files, masker, mask_data, working_folder):
#    for file in files:
#        pdata = nib.load(os.path.join(working_folder, file)).get_data()
#        pvals = pdata[mask_data > 0]
#        z_map = masker.inverse_transform(
#            z_score(np.power(10, -np.abs(pvals))))
#        thresholded_map2, threshold2 = map_threshold(
#            z_map, threshold=.05, height_control='fdr')
#        print(file, threshold2)
#
#        if 'display' in locals():
#            plotting.plot_stat_map(
#                thresholded_map2,
#                cut_coords=display.cut_coords,
#                title='Thresholded z map, expected fdr = .05',
#                threshold=threshold2)
#        else:
#            display = plotting.plot_stat_map(
#                thresholded_map2,
#                title='Thresholded z map, expected fdr = .05',
#                threshold=threshold2)
#
#
#print_images(diagnosis_files, nifti_masker, mask_data, working_folder)
#print_images(age_files, nifti_masker, mask_data, working_folder)
#print_images(sex_files, nifti_masker, mask_data, working_folder)
