function [config, path_table] = NM_process(config, step, use_adjustments)
%--------------------------------------------------------------------------
% NuMorph processing pipeline designed to perform channel alignment,
% intensity adjustment, and stitching on multi-channel light-sheet images.
%
%--------------------------------------------------------------------------
% Usage:  
% [config, path_table] = NM_process(config, step, use_adjustments)
% 
%--------------------------------------------------------------------------
% Inputs:
% config: Config structure for processing.
% 
% step: ('process','stitch','align','intensity'). Perform a single
% processing or run full pipeline. (default: 'process')
%   
% use_adjustments: Apply intensity adjustments if performing 1 step. 
% (default: true)
%
%--------------------------------------------------------------------------
% Output:
% config: Updated config structure after processing.
% 
% path_table: Updated file path table after processing.
% 
%--------------------------------------------------------------------------

% If first input is string or char, generate config structure
if ischar(config) || isstring(config)
    config = NM_config('process',char(config));
end
config.home_path = fileparts(which('NM_config'));

% Check config structure to make sure it's correct
if ~isfield(config,'adjust_intensity') || ~isfield(config,'align_channels')
    error("config structure is not from processing template.")
end

% Default to run full pipeline
if nargin<2
    step = 'process';
end

% Apply intensity adjustments if only aligning or stitching
if nargin<3 
    if ~isequal(config.adjust_intensity,"false")
        use_adjustments = true;
    else
        use_adjustments = false;
    end
elseif ~use_adjustments
    config.adjust_intensity = "false";
end

% Check step input
step = char(step);
if ~ismember({step},{'process','align','stitch','intensity'})
    error("Invalid processing step specified.")
end

% Make an output directory
if exist(config.output_directory,'dir') ~= 7
    mkdir(config.output_directory);
end

% Make a variables directory
config.var_directory = fullfile(config.output_directory,'variables');
if exist(config.var_directory,'dir') ~= 7
    mkdir(config.var_directory);
end

fprintf("%s\t Working on sample %s \n",datetime('now'),config.sample_id)

%% Create directories
% Update image directory if using processed images
if ~isequal(config.use_processed_images,"false")
    config.img_directory = fullfile(config.output_directory,config.use_processed_images);
    if ~exist(config.img_directory,'dir')
        error("Could not locate processed image directory %s\n",config.img_directory)
    end
    if isequal(config.use_processed_images,"aligned")
        warning("Images already aligned. Skipping channel alignment")
        config.channel_alignment = "false";
    end
end

%% Read image filename information
% Generate table containing image information
path_table = path_to_table(config);

% Count number of x,y tiles for each channel
nchannels = length(unique(path_table.channel_num));
ncols = zeros(1,nchannels); nrows = zeros(1,nchannels);
for i = 1:nchannels
    path_sub = path_table(path_table.channel_num == i,:);
    ncols(i) = length(unique(path_sub.x));
    nrows(i) = length(unique(path_sub.y));
end

% Check if all resolutions are equal
equal_res = all(cellfun(@(s) config.resolution{1}(3) == s(1,3),config.resolution));

%% Intensity Adjustment
% Configure intensity adjustments
if use_adjustments || isequal(step,'intensity')
    % If single step, will only be loading intensity adjustments
    if isequal(step,'stitch') || isequal(step,'align')
        config.adjust_intensity = "true";
    end
    [config, path_table] = perform_intensity_adjustment(config, path_table, nrows, ncols);
end

%% Run single step and return if specified
% Note no specific checks on tiles/markers included
if nargin>1 && isequal(step,'stitch')    
    [config, path_table] = perform_stitching(config, path_table);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
elseif nargin>1 && isequal(step,'align')
    [config, path_table] = perform_channel_alignment(config, path_table, equal_res);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
elseif nargin>1 && isequal(step,'intensity')
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
end

%% Run multiple steps
% If equal numbers of column and row tiles, perform alignment on each tile,
% then do stitching on aligned images. This assumes the same resolution for
% each channel. If the channels are imaged at different resolutions, stitch
% each channel if multi-tile. Then run alignment
if all(ncols == max(ncols)) && all(nrows == max(nrows))
    % Equal number of tiles
    % Channel Alignment
    [config, path_table] = perform_channel_alignment(config,path_table,equal_res);
    
    % Stitching
    if ncols(1)*nrows(1) == 1
        fprintf("%s\t 1 tile detected for marker %s \n",datetime('now'),config.markers(1));
    else
        [config, path_table] = perform_stitching(config, path_table);
    end
else
    % Different number of tiles
    fprintf("%s\t Different number of tiles between channels \n",datetime('now'))

    % Check resolutions are present
    assert(length(config.resolution) == length(config.markers),"Specify resolution "+...
        "for each channel as cell array");
    
    % Stitching
    for i = 1:length(config.markers)
        % Continue if only 1 tile
        if ncols(i)*nrows(i) == 1
            fprintf("%s\t 1 tile detected for marker %s \n",datetime('now'),config.markers(i));
            continue
        else
            config.stitch_sub_channel = i;
        end
        [config, path_table] = perform_stitching(config, path_table);
    end
    
    % Channel Alignment
    % First make sure there's only 1 tile in each channel
    assert(max(path_table.x) == 1,...
        "More than 1 tile detected for marker %s. Images "+...
        "must be stitched first when multiple resolutions are present.",...
        config.markers(ncols == max(ncols)))
    assert(max(path_table.y) == 1,...
        "More than 1 tile detected for marker %s. Images "+...
        "must be stitched first when multiple resolutions are present.",...
        config.markers(nrows == max(nrows)))
    
    [config, path_table] = perform_channel_alignment(config, path_table, equal_res);
end

%% Return outputs
if nargout == 1; path_table = []; end
if nargout < 1; config = []; end
    
end
