function [path_table_series, path_table_nii] = munge_string(paths)
% Try to create path table for single folder with whatever information is
% available.

% These are all possible image extensions we're able to read
ext = [".tif",".tiff",".nii",".nrrd",".nhdr",".mhd"];

% Default position expression vector
position_exp = ["[\d*", "\d*]","Z\d*"];

% Check if there are .tifs or .niis or both
all_files = dir(paths);

% Get which image extensions are present and subset these images
idx = false(length(ext),length(all_files));
for i = 1:length(ext)
    idx(i,:) = arrayfun(@(s) contains(s.name,ext(i)),all_files);
end
ext_here = ext(any(idx,2));

%%%%%%%%%%%%%% Need to update
%%%%%%%%%%%%%% Read nifti and import resolution
% Read nifiti file formats if present
nifti_ext = ext_here(~ismember(ext_here,[".tif",".tiff"]));
if ~isempty(nifti_ext)
    nifti_files = all_files(any(idx(3:end,:)));
    path_table_nii = munge_nifti_string(nifti_files);
else
    path_table_nii = [];
end

% Subset tif files
tiff_files = all_files(any(idx(1:2,:)));

% Fill in table
path_table_series = table('Size',[length(tiff_files), 7],...
    'VariableTypes',{'cell','string','string','string','double','double','double'},...
    'VariableNames',{'file','sample_id','markers','channel_num','x','y','z'});
path_table_series.file = arrayfun(@(s) fullfile(s.folder,s.name),tiff_files,'UniformOutput',false);
names = {tiff_files.name};

% Can only do channel number here
path_table_series.channel_num = cellfun(@(s) str2double(regexprep(regexp(s,'_C\d*_','match'),{'_C','_'},'')),names)';

% Do slice positions
try
    path_table_series.y = cellfun(@(s) str2double(regexprep(regexp(s,position_exp(1),'match'),'[^\d+]','')),names)';
    path_table_series.x = cellfun(@(s) str2double(regexprep(regexp(s,position_exp(2),'match'),'[^\d+]','')),names)';
catch
    path_table_series.y = ones(height(path_table_series),1);
    path_table_series.x = ones(height(path_table_series),1);
end

try
    path_table_series.z = cellfun(@(s) str2double(regexprep(regexp(s,position_exp(3),'match'),'[^\d+]','')),names)';    
catch
    a = arrayfun(@(s) regexp(s.name,'\d*','Match'), paths, 'UniformOutput', false);
    a = cat(1,a{:});
    idx = false(1,size(a,2));
    for i = size(a,2)
        idx(i) = ~all(a{1,i} == a{end,i});
    end
    
    if sum(idx) == 1
        path_table_series.z = str2double(a(:,idx));
    else
        path_table_series.z = ones(size(a,1),1);
    end
end

% Set channels and positions to start at 1
path_table_series.channel_num = path_table_series.channel_num - min(path_table_series.channel_num) + 1;
path_table_series.x = path_table_series.x- min(path_table_series.x) + 1;
path_table_series.y = path_table_series.y - min(path_table_series.y) + 1;
path_table_series.z = path_table_series.z - min(path_table_series.z) + 1;

% Fill in remainig columns
path_table_series.sample_id = repmat("SAMPLE",height(path_table_series),1);
path_table_series.markers = "marker"+num2str(path_table_series.channel_num);

end


function paths_nifti = munge_nifti_string(nifti_files)
% Create structure for nifiti files. Assumed to single tile, single
% channel, multi-slice. Must contain a marker in the filename

file = cell(1,length(nifti_files));
marker = repmat("",1,length(nifti_files));
channel_num = zeros(1,length(nifti_files));
y_res = zeros(1,length(nifti_files));
x_res = zeros(1,length(nifti_files));
z_res = zeros(1,length(nifti_files));

% Subset only images with marker present
idx=1;
for i = 1:length(config.mri_markers)
    sub = nifti_files(arrayfun(@(s) contains(s.name,config.mri_markers(i)),nifti_files));
    for j = 1:length(sub)
       file{idx} = fullfile(sub(j).folder,sub(j).name);
       marker(idx) = config.mri_markers(i);
       channel_num(idx) = i;
       y_res(idx) = config.mri_resolution(1);
       x_res(idx) = config.mri_resolution(2);
       z_res(idx) = config.mri_resolution(3);
       idx = idx+1;
    end
end

% Create table
paths_nifti = table('Size',[length(1:idx-1), 7],...
    'VariableTypes',{'cell','string','string','string','double','double','double'},...
    'VariableNames',{'file','sample_id','marker','channel_num','x_res','y_res','z_res'});

paths_nifti.file = file(1:idx-1)';
paths_nifti.sample_id = repmat(config.sample_id,idx-1,1);
paths_nifti.marker = marker(1:idx-1)';
paths_nifti.channel_num = channel_num(1:idx-1)';
paths_nifti.y_res = y_res(1:idx-1)';
paths_nifti.x_res = x_res(1:idx-1)';
paths_nifti.z_res = z_res(1:idx-1)';

end
