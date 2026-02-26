function resampled_image = resample_path_table(path_table, config, channels, resolution, resample_res)
%--------------------------------------------------------------------------
% Resample image to specific resolution. 
%--------------------------------------------------------------------------
% Inputs:
% path_table - table of image paths. Images should contain only 1 tile.
%
% config - config structure from NM_analyze. Optional: can leave empty and
% specify resoltion and resample_res.
%
% resolution - [x,y,z] specifying input image resolution. Can also send as
% cell array of resolutions if different for each channel. 
%
% resample_res - [x,y,z] resolution to resample to.
%
% channels - (default: length) index specifying which channels to resample.
%--------------------------------------------------------------------------

% Must specify resample resolution somewhere
if isempty(config) && nargin<3
    error("Must provide config structure or specify resolutions.")
end

% Default do all channels in path_table
if nargin<3
    channels = unique(path_table.channel_num);
end

% Default resample resolution
if nargin<4
    resolution = config.resolution;
    resample_res = config.resample_resolution;
end
    
% Check resolution
if ~iscell(resolution)
    resolution = repmat({resolution},1,length(channels));
end

% Resample image name resolution
if length(resample_res) == 1
    resample_res = repmat(resample_res,1,3);
    res_name = string(resample_res(1));
else
    res_name = string(resample_res);
end

% Create resampled directory
resample_path = fullfile(config.output_directory,'resampled');
if ~isfolder(resample_path)
    mkdir(resample_path)
end

% Perform resampling for 1 or multiple channels
nchannels = length(channels);
resampled_image = cell(1,nchannels);
for i = 1:nchannels
    fprintf('%s\t Resampling channel %d\n',datetime('now'),channels(i))            
    path_sub = path_table(path_table.channel_num == channels(i),:);
    nb_images = height(path_sub);

    % Measure image dimensions for high resolution image
    tempI = imread(path_sub.file{1});
    [nrows,ncols] = size(tempI);

    % Calculate image dimensions for target resolution
    re_v = resample_res./resolution{channels(i)};
    re_height = round(nrows/re_v(1));
    re_width = round(ncols/re_v(2));
    re_z = round(nb_images/re_v(3));

    % Resample first only in x,y to decrease memory demands
    fprintf('Reading and resampling XY\n')            
    re_I = repmat({zeros(re_height,re_width,'uint16')},1,nb_images);
    for j = 1:nb_images
        I = loadtiff(path_sub.file{j});
        re_I{j} = imresize(I,[re_height,re_width]);
    end
    re_I = cat(3,re_I{:});

    % Then resample in z 
    fprintf('Resampling Z\n')            
    re_I = imresize3(re_I,[re_height,re_width,re_z]);

    % Save as nii series
    marker = path_table(path_table.channel_num == channels(i),:).markers{1};
    if ~isempty(config) && isfield(config,'sample_id')
        filepath = fullfile(resample_path,sprintf('%s_C%d_%s_%s.nii',config.sample_id,...
            channels(i),marker,res_name));
    else
        filepath = fullfile(resample_path,sprintf('C%d_%s_%s.nii',...
            channels(i),marker,res_name));
    end
    niftiwrite(re_I, filepath)
    
    % Check if providing resampled image
    if nargout == 1
        resampled_image{i} = re_I;
    end
end

end