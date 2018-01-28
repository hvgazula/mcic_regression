#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:26:52 2018

@author: Harshvardhan Gazula
@acknowledgments: Eswar Damaraju for showing the relevant MATLAB commands
@notes: Contains code to extract information from the MCIC NIfTI files
"""

import os
import pickle
import nibabel as nib
import numpy as np
import pandas as pd


def nifti_to_data(image_files_location, mask_array):
    """Extracts data from the nifti image_files

    Args:
        image_files_location (string): Path to the folder where the images are
                                        stored
        mask_array (array): Array from Mask file for processing the nii files

    Returns:
        List of numpy arrays from each NIfTI file

    """
    appended_data = []

    # List all files
    image_files = sorted([
        f for f in os.listdir(image_files_location)
        if os.path.isfile(os.path.join(image_files_location, f))
    ])

    # Extract Data (after applying mask)
    for image in image_files:
        print(image)
        image_data = nib.load(
            os.path.join(image_files_location, image)).get_data()
        appended_data.append(image_data[mask_array > 0])

    return appended_data


def extract_demographic_info(folder_string):
    """Extract demographic information from csv files

    Args:
        folder_string (string): Path to folder with demographic information

    Returns:
        DataFrame with the demographic information

    """
    demographic_info = pd.read_csv(folder_string, delimiter=',')
    demographic_info = demographic_info.drop('subject_id', axis=1)

    return demographic_info


data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask')
demographics_location = os.path.join(data_location, 'demographics')
patient_image_files_location = os.path.join(data_location, 'group1_patients')
control_image_files_location = os.path.join(data_location, 'group2_controls')

# %% Extracting Mask Data
mask_file = os.path.join(mask_location, 'mask.nii')
mask_data = nib.load(mask_file).get_data()

# %% Folder specific to Patients and Controls *.nii files
patient_image_files_location = os.path.join(data_location, 'group1_patients')
control_image_files_location = os.path.join(data_location, 'group2_controls')

# %% Reading Voxel Info into an Array
print('Extracting Info from NIFTI files')
patient_data = nifti_to_data(patient_image_files_location, mask_data)
control_data = nifti_to_data(control_image_files_location, mask_data)

voxels = pd.DataFrame(np.vstack((patient_data, control_data)))

# %% (Redundant) Reading demographic information
patient_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'patients.csv'))
control_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'controls.csv'))

# %% Replacing all diagnosis values with either 'Patient' or 'Control'
patient_demographics['diagnosis'] = 'Patient'
control_demographics['diagnosis'] = 'Control'

demographics = pd.concat(
    [patient_demographics, control_demographics], axis=0).reset_index(drop=True)

# %% Writing to a file
print('writing data to a pickle file')
with open("final_data.pkl", "wb") as f:
    pickle.dump((demographics, voxels), f)

print('Finished Running Script')
