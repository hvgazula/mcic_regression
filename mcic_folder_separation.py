#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 08:33:59 2018

@author: hgazula
"""
from shutil import copy
import json
import os
import pandas as pd
from mcic_classes import CovarInfo, DataLocationInfo


def extract_demographic_info(csv_file):
    """Extract demographic information from csv files
    Args:
        destination_folder (string): Path to folder with demographic information
    Returns:
        DataFrame with the demographic information
    """
    demographic_info = pd.read_csv(csv_file, delimiter=',')

    return demographic_info


#pylint: disable=too-many-arguments
def load_covariate_strings(cinfo, value, destination, folder, demographics,
                           flag):
    """Returns a list with the covariate vector for each subject
    """
    inputspec_list = []

    image_files = return_files(folder)

    for file in image_files:
        if file.startswith(value):
            src = os.path.join(folder, file)
            copy(src, destination)
            curr_id = file.split('_')[0]
            covar_values = demographics.loc[demographics["subject_id"] ==
                                            curr_id, cinfo.
                                            covar_list[1:]].iloc[0].tolist()

            covar_values = [
                int(x) if not isinstance(x, str) else x for x in covar_values
            ]
            substring = [file, flag, *covar_values]
            inputspec_list.append(substring)

    return inputspec_list


def return_files(folder):
    """Returns contents of the folder after sorting
    """
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])


def write_local_inputspec(spec_file, spec_list, cinfo):
    """Template function to write inputspec.json"""
    with open(spec_file, 'w') as file_h:
        file_h.write('{')
        file_h.write('''"covariates": {
    "value": [\n''')
        file_h.write('''[[''')
        file_h.write('[' +
                     ','.join('"{}"'.format(x)
                              for x in ['niftifile', *cinfo.covar_list]) +
                     '],\n')
        file_h.write(',\n'.join(json.dumps(x) for x in spec_list))
        file_h.write('''\n]],\n''')
        file_h.write('[' + ','.join('"{}"'.format(x)
                                    for x in cinfo.covar_list) + '],\n')
        file_h.write('[' + ','.join('"{}"'.format(x)
                                    for x in cinfo.covar_types) + ']\n')

        file_h.write(']\n')
        file_h.write('},\n')
        file_h.write('''"data": {
    "value": [\n''')
        file_h.write('\t [' + ',\n\t'.join('"{}"'.format(l[0])
                                           for l in spec_list) + '],\n')
        file_h.write('''[ "niftifile" ]\n''')
        file_h.write('] \n },\n')
        file_h.write('''"lambda": {
    "value": 0 \n''')
        file_h.write('''}\n''')
        file_h.write('''}\n''')


def main():
    """This is where the program starts
    """
    covar_info = CovarInfo(0)
    path_info = DataLocationInfo()

    patient_demographics = extract_demographic_info(
        os.path.join(path_info.demographics_location,
                     path_info.patients_csv_file))
    control_demographics = extract_demographic_info(
        os.path.join(path_info.demographics_location,
                     path_info.controls_csv_file))

    if not os.path.exists(path_info.output_folder):
        os.makedirs(path_info.output_folder)

    site_list = ['M02', 'M52', 'M55', 'M87']

    data = []
    for index, site_id in enumerate(site_list):
        destination_folder = os.path.join(path_info.output_folder,
                                          'local{:01d}'.format(index),
                                          'simulatorRun')
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        patient_spec = load_covariate_strings(covar_info, site_id,
                                              destination_folder,
                                              path_info.patient_images_folder,
                                              patient_demographics, False)
        control_spec = load_covariate_strings(covar_info, site_id,
                                              destination_folder,
                                              path_info.control_images_folder,
                                              control_demographics, True)

        inputspec_list = [*patient_spec, *control_spec]

        inputspec_file = os.path.join(destination_folder,
                                      'inputspec{:02d}.json'.format(index))
        write_local_inputspec(inputspec_file, inputspec_list, covar_info)

        with open(inputspec_file, 'rb') as file_h:
            data.append(json.load(file_h))

    global_inputspec_file = os.path.join(path_info.output_folder,
                                         'inputspec.json')
    with open(global_inputspec_file, 'w') as fwrite_h:
        json.dump(data, fwrite_h, indent=1)


if __name__ == '__main__':
    main()
