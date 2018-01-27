#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:16:57 2018

@author: Harshvardhan Gazula
@acknowledgments: Eswar Damaraju (for providing the relevant AFNI commands)
@notes: Contains code to resample the NIfTI images
    http://nipype.readthedocs.io/en/latest/interfaces/generated/interfaces.afni/utils.html#resample
"""

import os
from nipype.interfaces import afni


def resample_nifti_images(images_location, voxel_dimensions, resample_method):
    """Resample the NIfTI images in a folder
    Arguments:
        images_location: Path where the images are stored
        voxel_dimension: tuple (dx, dy, dz)
        resample_method: NN - Nearest neighbor
                         Li - Linear interpolation

    Returns:
        None: But puts the resampled *.nii files in a new folder
    """
    image_files = sorted([
        f for f in os.listdir(images_location)
        if os.path.isfile(os.path.join(images_location, f))
    ])

    folder_tag = '_resampled'
    new_folder = images_location + folder_tag

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for image_file in image_files:
        print(image_file)
        (file_name, file_ext) = os.path.splitext(image_file)
        new_file_name = file_name + '_4mm' + file_ext

        resample = afni.Resample()
        resample.inputs.environ = {'AFNI_NIFTI_TYPE_WARN': 'NO'}
        resample.inputs.in_file = os.path.join(images_location, image_file)
        resample.inputs.out_file = os.path.join(new_folder, new_file_name)
        resample.inputs.voxel_size = voxel_dimensions
        resample.inputs.outputtype = 'NIFTI'
        resample.inputs.resample_mode = resample_method
        resample.run()


data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
mask_location = os.path.join(data_location, 'mask')
patient_images_location = os.path.join(data_location, 'group1_patients')
control_images_location = os.path.join(data_location, 'group2_controls')

voxel_size = (4.0, 4.0, 4.0)

resample_nifti_images(mask_location, voxel_size, 'NN')
resample_nifti_images(patient_images_location, voxel_size, 'Li')
resample_nifti_images(control_images_location, voxel_size, 'Li')
