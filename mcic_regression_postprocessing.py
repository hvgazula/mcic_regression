#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:52:22 2018

@author: Harshvardhan Gazula
@notes: Contains code to process the dataframes from each regression
        and put write them to *.nii file format
"""

import os
import shelve
import nibabel as nib
import numpy as np

folder_index = input('Enter the Folder Name to perform postprocessing: ')
folder_name = folder_index.replace(' ', '_')

images_folder = os.path.join(folder_name, 'nifti_files')
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# %% Head Directory where the data is stored and loading the mask
data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask_resampled')
mask = nib.load(os.path.join(mask_location, 'mask_4mm.nii'))

# %% Converting the data to NIfTI format
for file in os.listdir(folder_name):
    if file.endswith('.dat'):
        print(file)
        my_dict = shelve.open(
            os.path.join(folder_name, os.path.splitext(file)[0]))
        for key in my_dict:
            print(key)
            for column in my_dict[key].columns.tolist():
                image_string_sequence = (
                    key, column, file.split('_')[0] + file.split('_')[2])
                image_string = '_'.join(image_string_sequence) + '.nii'
                new_data = np.zeros(mask.shape)
                if key == 'pvalues':
                    new_data[mask.get_data() >
                             0] = -1 * np.log10(my_dict[key][
                                 column]) * np.sign(my_dict['tvalues'][column])
                clipped_img = nib.Nifti1Image(new_data, mask.affine,
                                              mask.header)
                print('Saving ', image_string)
                nib.save(clipped_img, os.path.join(images_folder,
                                                   image_string))

        my_dict.close()
