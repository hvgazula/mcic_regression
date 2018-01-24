function mcic_regression_print_images()
% Writen: 01/11/2018
% Author: Harshvardhan Gazula
% Acknowledgments: Eswar Damaraju
% Notes: This file contains script to generate images from the .nii files
% Steps:
% 1) Assume the *.nii files are in one folder
% 2) If the template .nii does not exist copy it into the above folder
% 3) The p-value in .nii files contain (-log10(p-value)*t-value)
% 4) The script has been written in such a way that the slice coordinates
% for the decentralized images are same as their centralized counterparts

working_folder = input('Please Enter the Folder with output NIfTI files: ', 's');
working_folder = fullfile(pwd, working_folder);

template_file = fullfile(working_folder, 'MNI152_T1_1mm_brain.nii');

if exist(fullfile(working_folder, 'MNI152*.nii'), 'file') ~= 2
    shell_cmd = ['gunzip -c ' ...
        '/export/apps/linux-x86_64/fsl/fsl-5.0.5/data/standard/MNI152_T1_1mm_brain.nii.gz ' ...
        '> ' template_file];
    system(shell_cmd);
end

% File Path based on Operating System
if ispc
    dirFlag = 'T:';
    
    GIFT_Path 	= '/apps/linux-x86/matlab/toolboxes/GroupICATv4.0a';
    SPM_Path 	= '/apps/linux-x86/matlab/toolboxes/spm12';
    MCI_Path 	= '/apps/linux-x86/matlab/toolboxes/MCIv4';
    FNC_Path 	= '/apps/linux-x86/matlab/toolboxes/FncVer2.3/';
    
    addpath(genpath(fullfile(dirFlag, GIFT_Path)))
    addpath(genpath(fullfile(dirFlag, SPM_Path)))
    addpath(genpath(fullfile(dirFlag, MCI_Path)))
    addpath(genpath(fullfile(dirFlag, FNC_Path)))
else
    dirFlag = '/export';
end

DataStored = fullfile(dirFlag, '/mialab/users/spanta/MCIC_2sample_ttest');

% Mask Location
Mask = fullfile(DataStored, 'outputs/outputs_default_options/mask.nii');

% Extract data relevant to regressors (age and diagnosis)
diagnosis_files = dir(fullfile(working_folder, 'pvalues_diagnosis_Patient_*.nii'));
age_files 		= dir(fullfile(working_folder, 'pvalues_age_*.nii'));
sex_files 		= dir(fullfile(working_folder, 'pvalues_sex_*.nii'));

% Add function:export_fig to the path
addpath(fullfile(dirFlag,'mialab','users','eswar','software','export_fig'));

% Printing Images
print_images(diagnosis_files, template_file, Mask, working_folder)
print_images(age_files, template_file, Mask, working_folder)
print_images(sex_files, template_file, Mask, working_folder)

function print_images(files, template_file, Mask, working_folder)

for i = 1 : length(files)
    disp(files(i).name)
    
    current_file    = fullfile(working_folder, files(i).name);
    p               = icatb_read_data(current_file, [], Mask)';
    pID             = fnctb_fdr(10.^-abs(p), 0.05);
    [fname, ~]      = mci_interp2struct(current_file, 1, template_file);
    
    if exist('slices', 'var')
        [slices, ~] = mci_makeimage(fname, template_file, 1,...
            'slicemethod'   , slices, ...
            'threshold_low' , -log10(pID),...
            'units'         , '-log_1_0 (p-value)');
    else
        [slices, ~] = mci_makeimage(fname, template_file, 1, ...
            'threshold_low' , -log10(pID),...
            'units'         , '-log_1_0 (p-value)');
    end
    [~, curr_file_name] = fileparts(current_file);
    eval(['export_fig ' fullfile(working_folder, curr_file_name) '.png  -nocrop']);
    
end