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
from mcic_classes import DataLocationInfo


def resample_nifti_images(images_location, voxel_dimensions, resample_method):
    """Resample the NIfTI images in a folder and put them in a new folder

    Args:
        images_location: Path where the images are stored
        voxel_dimension: tuple (dx, dy, dz)
        resample_method: NN - Nearest neighbor
                         Li - Linear interpolation

    Returns:
        None:

    """
    image_files = sorted([
        f for f in os.listdir(images_location)
        if os.path.isfile(os.path.join(images_location, f))
    ])

    voxel_size_str = '_{:.0f}mm'.format(float(voxel_dimensions[0]))
    new_folder = images_location + voxel_size_str

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for image_file in image_files:
        print(image_file)
        (file_name, file_ext) = os.path.splitext(image_file)
        new_file_name = ''.join([file_name, voxel_size_str, file_ext])

        resample = afni.Resample()
        resample.inputs.environ = {'AFNI_NIFTI_TYPE_WARN': 'NO'}
        resample.inputs.in_file = os.path.join(images_location, image_file)
        resample.inputs.out_file = os.path.join(new_folder, new_file_name)
        resample.inputs.voxel_size = voxel_dimensions
        resample.inputs.outputtype = 'NIFTI'
        resample.inputs.resample_mode = resample_method
        resample.run()


if __name__ == '__main__':
    VOXEL_SIZE = (4.0, 4.0, 4.0)
    DATA_INFO = DataLocationInfo()

    resample_nifti_images(DATA_INFO.mask_location, VOXEL_SIZE, 'NN')
    resample_nifti_images(DATA_INFO.patient_images_folder, VOXEL_SIZE, 'Li')
    resample_nifti_images(DATA_INFO.control_images_folder, VOXEL_SIZE, 'Li')
