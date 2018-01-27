#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:26:52 2018

@author: Harshvardhan Gazula
@notes: Contains code to extract information from the MCIC NIfTI files
"""
import os
import pickle
import nibabel as nib
import numpy as np
import pandas as pd


def nifti_to_data(ImagesLocation, MaskData):
    """Extracts data from the nifti images
    Args:
        ImagesLocation:
        MaskData:

    Returns:
        List of numpy arrays from each NIfti file

    """
    appended_data = []

    # List all files
    Images = sorted([
        f for f in os.listdir(ImagesLocation)
        if os.path.isfile(os.path.join(ImagesLocation, f))
    ])

    # Extract Data (after applying mask)
    for image in Images:
        print(image)
        ImageData = nib.load(
            os.path.join(ImagesLocation, image)).get_data()
        appended_data.append(ImageData[MaskData > 0])

    return appended_data


def extract_demographic_info(folder_string):
    """Extract demograhic information from csv files

    Args:
        folder_string

    Returns:
        DataFrame

    """
    Demo_Info = pd.read_csv(folder_string, delimiter=',')
    Demo_Info = Demo_Info.drop('subject_id', 1)

    return Demo_Info


data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask')
demographics_location = os.path.join(data_location, 'demographics')
patient_images_location = os.path.join(data_location, 'group1_patients')
control_images_location = os.path.join(data_location, 'group2_controls')

# %% Mask Location
mask_file = os.path.join(mask_location, 'mask.nii')
mask_data = nib.load(mask_file).get_data()

# %% Folder specific to Patients and Controls
patient_images_location = os.path.join(data_location, 'group1_patients')
control_images_location = os.path.join(data_location, 'group2_controls')

# %% Read Voxel Info into an Array
print('Extracting Info from NIFTI files')
patient_data = nifti_to_data(patient_images_location, mask_data)
control_data = nifti_to_data(control_images_location, mask_data)

voxels = pd.DataFrame(np.vstack((patient_data, control_data)))

# %% (Redundant) Read the demographic information
patient_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'patients.csv'))
control_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'controls.csv'))

# %% Replacing all diagnosis values with either 'patient' or 'control'
patient_demographics['diagnosis'] = 'Patient'
control_demographics['diagnosis'] = 'Control'

demographics = pd.concat(
    [patient_demographics, control_demographics], axis=0).reset_index(drop=True)

# %% write to a file
print('writing data to a pickle file')
with open("final_data.pkl", "wb") as f:
    pickle.dump((demographics, voxels), f)

print('Finished Running Script')
