#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 20:26:52 2018

@author: Harshvardhan Gazula
@notes: Contains code to extract information from the MCIC NIfTI files
"""
import os
import pickle
import shelve
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


DIR_FLAG = '/export/mialab'

# %% Head Directory where the data is stored
PATH_TO_DATA = os.path.join(DIR_FLAG, 'users', 'spanta', 'MCIC_2sample_ttest')

# %% Mask Location
Mask = os.path.join(PATH_TO_DATA, 'outputs/outputs_default_options/mask.nii')
MaskData = nib.load(Mask).get_data()

# %% Folder specific to Patients and Controls
PatientImagesLocation = os.path.join(PATH_TO_DATA, 'group1_patients')
ControlImagesLocation = os.path.join(PATH_TO_DATA, 'group2_controls')

# %% Read Voxel Info into an Array
print('Extracting Info from NIFTI files')
PatientData = nifti_to_data(PatientImagesLocation, MaskData)
ControlData = nifti_to_data(ControlImagesLocation, MaskData)

voxels = pd.DataFrame(np.vstack((PatientData, ControlData)))

# %% (Redundant) Read the demographic information
PatientDemo = extract_demographic_info(
    os.path.join(PATH_TO_DATA, 'demographics/patients.csv'))
ControlDemo = extract_demographic_info(
    os.path.join(PATH_TO_DATA, 'demographics/controls.csv'))

# %% Replacing all diagnosis values with either 'patient' or 'control'
PatientDemo['diagnosis'] = 'Patient'
ControlDemo['diagnosis'] = 'Control'

demographics = pd.concat(
    [PatientDemo, ControlDemo], axis=0).reset_index(drop=True)

# %% write to a file
print('writing data to a shelve file')
final_data = shelve.open('final_data')
final_data['demographics'] = demographics
final_data['voxels'] = voxels
final_data.close()

print('writing data to a pickle file')
with open("final_data.pkl", "wb") as f:
    pickle.dump((demographics, voxels), f)
