close all; clear; clc

addpath('/export/mialab/users/eswar/software/export_fig');
files = dir('*age_beta*.nii');

working_folder = pwd;
template_file = fullfile(working_folder, 'MNI152_T1_1mm_brain.nii');

if exist(fullfile(working_folder, 'MNI152*.nii'), 'file') ~= 2
    shell_cmd = ['gunzip -c ' ...
        '/export/apps/linux-x86_64/fsl/fsl-5.0.5/data/standard/MNI152_T1_1mm_brain.nii.gz ' ...
        '> ' template_file];
    system(shell_cmd);
end

data_location = '/export/mialab/users/hgazula/mcic_regression/mcic_data';
mask_location = fullfile(data_location, 'mask');

% Mask Location
Mask = fullfile(mask_location, 'mask.nii');

CM = colormap('parula');
save colormapParula CM

for i = 1 : length(files)
    current_file    = fullfile(working_folder, files(i).name);
    p               = icatb_read_data(current_file, [], Mask)';
    p_vec(:, i) = (p);
end

p_thresholdh = max(max(abs(p_vec)));
 
for i = 1 : length(files)
    disp(files(i).name)
    
    current_file    = fullfile(working_folder, files(i).name);  
    [fname, ~]      = mci_interp2struct(current_file, 1, template_file);
    
    if exist('slices', 'var')
        [slices, ~] = mci_makeimage(fname, template_file, 1, ...                         
            'units', '\beta_{Age}', ...
        'absflag', 1, ...
        'cmfile', 'colormapParula', ...
        'slicemethod', slices);
    else
        [slices, ~] = mci_makeimage(fname, template_file, 1, ...             
            'units', '\beta_{Age}', ...
            'absflag', 1, ...
            'cmfile', 'colormapParula');
    end
    
    [~, curr_file_name] = fileparts(current_file);
    eval(['export_fig ' fullfile(working_folder, curr_file_name) '.png  -nocrop']);
    
end