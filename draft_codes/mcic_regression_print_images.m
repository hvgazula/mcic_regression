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
working_folder = fullfile(pwd, working_folder, 'images');

template_file = fullfile(working_folder, 'MNI152_T1_1mm_brain.nii');

if exist(fullfile(working_folder, 'MNI152*.nii'), 'file') ~= 2
    shell_cmd = ['gunzip -c ' ...
        '/export/apps/linux-x86_64/fsl/fsl-5.0.5/data/standard/MNI152_T1_1mm_brain.nii.gz ' ...
        '> ' template_file];
    system(shell_cmd);
end

data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data';
mask_location = fullfile(data_location, 'mask_resampled');

% Mask Location
Mask = fullfile(mask_location, 'mask_4mm.nii');

% Extract data relevant to regressors (age and diagnosis)
diagnosis_files = dir(fullfile(working_folder, 'pvalues_diagnosis_Patient_*.nii'));
age_files 		= dir(fullfile(working_folder, 'pvalues_age_*.nii'));
sex_files 		= dir(fullfile(working_folder, 'pvalues_sex_*.nii'));

% Add function:export_fig to the path
addpath('/export/mialab/users/eswar/software/export_fig');

% Printing Images
print_images(diagnosis_files, template_file, Mask, working_folder)
print_images(age_files, template_file, Mask, working_folder)
print_images(sex_files, template_file, Mask, working_folder)

function print_images(files, template_file, Mask, working_folder)

for i = 1 : length(files)
    disp(files(i).name)
    
    for i = 1 : length(files)
        current_file    = fullfile(working_folder, files(i).name);
        p[:,1]          = icatb_read_data(current_file, [], Mask)';
    end
    
    thresholdh = max(max(abs(p)));
    thresholdhl = max(max(abs(p))) * 0.5;
    
    
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
    eval(['export_fig ' fullfile(working_folder, '..', 'processed_images', curr_file_name) '.png  -nocrop']);
    
end