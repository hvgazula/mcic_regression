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


def extract_demographic_info(csv_file):
    """Extract demographic information from csv files
    Args:
        destination_folder (string): Path to folder with demographic information
    Returns:
        DataFrame with the demographic information
    """
    demographic_info = pd.read_csv(csv_file, delimiter=',')

    return demographic_info


def load_covariate_strings(value, destination, folder, demographics, flag):
    """Returns a list with the covariate vector for each subject
    """
    cinfo = CovarInfo()

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


class CovarInfo():
    """Contains information related to covariates
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.data_location = '/Users/hgazula/Downloads/mcic_regression/mcic_data'
        self.demographics_folder_name = 'demographics'
        self.patients_folder_name = 'group1_patients_4mm'
        self.controls_folder_name = 'group2_controls_4mm'
        self.patients_csv_file = 'patients.csv'
        self.controls_csv_file = 'controls.csv'
        self.output_folder = 'coinstac_data4'
        self.num_extra_covars = 0

        self.demographics_location = None
        self.patient_images_folder = None
        self.control_images_folder = None

        self.extra_covars = None
        self.covar_list = None
        self.extra_covar_types = None
        self.covar_types = None

    def createpaths(self):
        """Created full paths to required folders
        """
        self.demographics_location = os.path.join(
            self.data_location, self.demographics_folder_name)
        self.patient_images_folder = os.path.join(self.data_location,
                                                  self.patients_folder_name)
        self.control_images_folder = os.path.join(self.data_location,
                                                  self.controls_folder_name)

    def create_covars(self):
        """Creates necessary details for extra covariates
        """
        self.extra_covars = [
            'var{}'.format(x) for x in range(1, self.num_extra_covars + 1)
        ]
        self.covar_list = ["isControl", "age", "sex", *self.extra_covars]

        self.extra_covar_types = ["number"] * self.num_extra_covars
        self.covar_types = [
            "boolean", "number", "string", *self.extra_covar_types
        ]


def main():
    """This is where the program starts
    """
    pathinfo = CovarInfo()

    patient_demographics = extract_demographic_info(
        os.path.join(pathinfo.demographics_location,
                     pathinfo.patients_csv_file))
    control_demographics = extract_demographic_info(
        os.path.join(pathinfo.demographics_location,
                     pathinfo.controls_csv_file))

    if not os.path.exists(pathinfo.output_folder):
        os.makedirs(pathinfo.output_folder)

    site_list = ['M02', 'M52', 'M55', 'M87']

    for index, site_id in enumerate(site_list):
        destination_folder = os.path.join(pathinfo.output_folder,
                                          '{:02d}'.format(index),
                                          'simulatorRun')
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        patient_spec = load_covariate_strings(site_id, destination_folder,
                                              pathinfo.patient_images_folder,
                                              patient_demographics, False)
        control_spec = load_covariate_strings(site_id, destination_folder,
                                              pathinfo.control_images_folder,
                                              control_demographics, True)

        inputspec_list = [*patient_spec, *control_spec]

        inputspec_file = os.path.join(destination_folder,
                                      'inputspec{:02d}.json'.format(index))

        with open(inputspec_file, 'w') as file_h:
            file_h.write('{')
            file_h.write('''"covariates": {
        "value": [\n''')
            file_h.write('''[[''')
            file_h.write('[' + ','.join(
                '"{}"'.format(x)
                for x in ['niftifile', *pathinfo.covar_list]) + '],\n')
            file_h.write(',\n'.join(json.dumps(x) for x in inputspec_list))
            file_h.write('''\n],]\n''')
            file_h.write('[' + ','.join('"{}"'.format(x)
                                        for x in pathinfo.covar_list) + '],\n')
            file_h.write('[' + ','.join('"{}"'.format(x)
                                        for x in pathinfo.covar_types) + ']\n')

            file_h.write(']\n')
            file_h.write('},\n')
            file_h.write('''"data": {
        "value": [\n''')
            file_h.write('\t [' + ',\n\t'.join('"{}"'.format(l[0])
                                               for l in inputspec_list) +
                         '],\n')
            file_h.write('''[ "niftifile" ]\n''')
            file_h.write('] \n },\n')
            file_h.write('''"lambda": {
        "value": 0 \n''')
            file_h.write('''}\n''')
            file_h.write('''}\n''')


if __name__ == '__main__':
    main()
