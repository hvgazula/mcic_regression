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


def resample_nifti_images(ImagesLocation):

    ImageFiles = sorted([
        f for f in os.listdir(ImagesLocation)
        if os.path.isfile(os.path.join(ImagesLocation, f))
    ])

    new_folder = ImagesLocation + '_' + folder_tag

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for ImageFile in ImageFiles:
        resample = afni.Resample()
        (file_name, file_ext) = os.path.splitext(ImageFile)
        new_file_name = file_name + '_4mm' + file_ext
        resample.inputs.in_file = os.path.join(ImagesLocation,
                                               ImageFile)
        resample.inputs.out_file = os.path.join(new_folder, new_file_name)
        resample.inputs.voxel_size = (dx, dy, dz)
        resample.inputs.outputtype = 'NIFTI'
        resample.inputs.resample_mode = 'NN'
        #    print(resample.cmdline)
        resample.run()


DataLocation = '/export/mialab/users/hgazula/mcic_regression/mcic_data'
(dx, dy, dz) = (4.0, 4.0, 4.0)

folder_tag = 'resampled'

MaskLocation = os.path.join(DataLocation, 'mask')
PatientImagesLocation = os.path.join(DataLocation, 'group1_patients')
ControlImagesLocation = os.path.join(DataLocation, 'group2_controls')

resample_nifti_images(MaskLocation)
resample_nifti_images(PatientImagesLocation)
resample_nifti_images(ControlImagesLocation)
