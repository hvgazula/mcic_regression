#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:36:28 2019

@author: hgazula
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataLocationInfo:
    """Contains information related to data paths
    """
    data_location: str = '/Users/hgazula/Downloads/mcic_regression/mcic_data'
    demographics_folder_name: str = 'demographics'
    patients_folder_name: str = 'group1_patients'
    controls_folder_name: str = 'group2_controls'
    patients_csv_file: str = 'patients.csv'
    controls_csv_file: str = 'controls.csv'
    output_folder: str = 'coinstac_data'

    demographics_location: str = ''
    patient_images_folder: str = ''
    control_images_folder: str = ''
    mask_location: str = ''

    def __post_init__(self):
        self.mask_location = os.path.join(self.data_location, 'mask')
        self.demographics_location = os.path.join(
            self.data_location, self.demographics_folder_name)
        self.patient_images_folder = os.path.join(self.data_location,
                                                  self.patients_folder_name)
        self.control_images_folder = os.path.join(self.data_location,
                                                  self.controls_folder_name)


@dataclass
class CovarInfo:
    """Contains information related to extra covariates"""
    num_extra_covars: int = 0
    extra_covars: List = field(default_factory=lambda: [])
    covar_list: List = field(default_factory=lambda: [])
    extra_covar_types: List = field(default_factory=lambda: [])
    covar_types: List = field(default_factory=lambda: [])

    def __post_init__(self):
        self.extra_covars = [
            'var{}'.format(x) for x in range(1, self.num_extra_covars + 1)
        ]
        self.covar_list = ["isControl", "age", "sex", *self.extra_covars]
        self.extra_covar_types = ["number"] * self.num_extra_covars
        self.covar_types = [
            "boolean", "number", "string", *self.extra_covar_types
        ]
