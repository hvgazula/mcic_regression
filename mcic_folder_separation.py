#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 08:33:59 2018

@author: hgazula
"""
import os
import pandas as pd
from shutil import copy


def extract_demographic_info(folder_string):
    """Extract demographic information from csv files
    Args:
        folder_string (string): Path to folder with demographic information
    Returns:
        DataFrame with the demographic information
    """
    demographic_info = pd.read_csv(folder_string, delimiter=',')
    #    demographic_info = demographic_info.drop('subject_id', axis=1)

    return demographic_info


data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask_resampled_3mm')
demographics_location = os.path.join(data_location, 'demographics')
patient_image_files_location = os.path.join(data_location,
                                            'group1_patients_resampled_3mm')
control_image_files_location = os.path.join(data_location,
                                            'group2_controls_resampled_3mm')

# List all files
patient_image_files = sorted([
    f for f in os.listdir(patient_image_files_location)
    if os.path.isfile(os.path.join(patient_image_files_location, f))
])

control_image_files = sorted([
    f for f in os.listdir(control_image_files_location)
    if os.path.isfile(os.path.join(control_image_files_location, f))
])

patient_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'patients.csv'))
control_demographics = extract_demographic_info(
    os.path.join(demographics_location, 'controls.csv'))

if not os.path.exists('coinstac_data'):
    os.makedirs('coinstac_data')

site_list = ['M02', 'M52', 'M55', 'M87']

for index, value in enumerate(site_list):
    inputspec_list = []
    folder_string = os.path.join('coinstac_data', '{:02d}'.format(index),
                                 'simulatorRun')
    if not os.path.exists(folder_string):
        os.makedirs(folder_string)

    # copy patient files
    for file in patient_image_files:
        if file.startswith(value):
            src = os.path.join(patient_image_files_location, file)
            copy(src, folder_string)
            curr_id = file.split('_')[0]
            curr_age = patient_demographics.loc[patient_demographics[
                "subject_id"] == curr_id, "age"].iloc[0]
            curr_sex = patient_demographics.loc[patient_demographics[
                "subject_id"] == curr_id, "sex"].iloc[0]
            substring = [file, "true", curr_age, curr_sex]
            inputspec_list.append(substring)

    # copy control files
    for file in control_image_files:
        if file.startswith(value):
            src = os.path.join(control_image_files_location, file)
            copy(src, folder_string)
            curr_id = file.split('_')[0]
            curr_age = control_demographics.loc[control_demographics[
                "subject_id"] == curr_id, "age"].iloc[0]
            curr_sex = control_demographics.loc[control_demographics[
                "subject_id"] == curr_id, "sex"].iloc[0]
            substring = [file, "false", curr_age, curr_sex]
            inputspec_list.append(substring)

    with open(
            os.path.join(folder_string, 'inputspec{:02d}.json'.format(index)),
            'w') as fh:
        fh.write('''{
  "covariates": {
    "value": [
      [
[["niftifile", "isControl", "age", "sex"],
''')
        for line in inputspec_list:
            fh.write("[\"%s\", %s, %d, \"%s\"],\n" % (line[0], line[1],
                                                      line[2], line[3]))
        fh.write(''' ] \n
      ],
      [
        "isControl",
        "age",
        "sex"
      ],
      [
        "boolean",
        "number",
        "string"
      ]
    ]
  },
  "data": {
    "value": [
      [ ''')
        for line in inputspec_list:
            fh.write("\"%s\",\n" % line[0])
        fh.write('''  ],
      [
        "niftifile"
      ]
    ]
  },
  "lambda": {
    "value": 0
  }
}''')
