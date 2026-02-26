function [config, path_table] = NM_analyze(config, step)
%--------------------------------------------------------------------------
% NuMorph analysis pipeline designed to perform image registration, nuclei
% counting, and cell-type classification based on nuclear protein markers
% in whole brain images.
%--------------------------------------------------------------------------
% Usage:  
% [config, path_table] = NM_analyze(config, step)
% 
% Inputs:
%   config - config structure for processing
%   step - 'stitch', 'align', 'intensity'. Perform only one of the 3 steps
%   use_adjustments - apply intensity adjustments if performing 1 step
%
% Output:
%   config - config structure after processing
%   path_table - table of filenames used and additional image information
%--------------------------------------------------------------------------

% Load configuration from .mat file, if not provided
if nargin<1
    config = load(fullfile('templates','NM_variables.mat'));
elseif isstring(config)
    config = load(fullfile(config,'NM_variables.mat'));
elseif ~isstruct(config)
    error("Invalid configuartion input")
end
config.var_directory = fullfile(config.output_directory,'variables');
config.home_path = fileparts(which('NM_config'));
config.res_name = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));

% Default to run full pipeline
if nargin<2
    step = 'analyze';
elseif ~ismember(step,{'analyze','resample','register','count','classify'})
    error("Unrecognized step selected")
end

% Make an output directory
if ~isfolder(config.output_directory)
    mkdir(config.output_directory);
end

% Make a variables directory
if ~isfolder(config.var_directory)
    mkdir(config.var_directory);
end

fprintf("%s\t Working on sample %s \n",datetime('now'),config.sample_id)

%% Read image filename information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate table containing image information
if isfield(config,'mri_directory') && ~isempty(config.mri_directory)
    [path_table, path_table_nii] = path_to_table(config, config.use_processed_images, false);
else
    path_table = path_to_table(config, config.use_processed_images, false);
    path_table_nii = [];
end

% If markers ignored, add these from raw
if ~isempty(path_table)
    if ~all(ismember(config.markers,unique(path_table.markers)))
        idx = find(~ismember(config.markers,unique(path_table.markers)));

        for i = 1:length(idx)
            config2 = config;
            config2.img_directory = config.img_directory(idx);
            config2.markers = config.markers(idx(i));
            config2.channel_num = config.channel_num(idx(i));
            try 
                path_table = vertcat(path_table,path_to_table(config2,'raw',false));
                path_table.channel_num = arrayfun(@(s) find(s == config.markers),path_table.markers);
            catch
                warning("Could not load marker %s ignored from processing.",config.markers(idx(i)));
            end
        end
        clear config2
    end
end

% Count number of x,y tiles for each channel
ntiles = length(unique([path_table.x,path_table.y]));
if ntiles ~= 1
    warning("Check if use_processed_images field needs to be updated to stitched directory")
    assert(ntiles == 1, "To perform analysis, there should be only 1 tile for "+...
        "each channel in the image dataset.")
end

%% Intialize results structure if not present
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
res_name = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));
if ~isfile(res_name)
    fprintf('%s\t Intializing new structure for saving results \n',datetime('now'))   
    results.sample_id = config.sample_id;
    results.group = config.group;
    results.sample_data.img_directory = config.img_directory;
    results.sample_data.output_directory = config.output_directory;
    results.sample_data.use_processed_images = config.use_processed_images;
    results.sample_data.resolution = config.resolution{1};
    results.sample_data.markers = config.markers;
    results.sample_data.hemisphere = config.hemisphere;
    results.sample_data.orientation = config.orientation;
    save(res_name,'-struct','results')
end

%% Run single step and return if specified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note no specific checks on tiles/markers included
if nargin>1 && isequal(step,'resample') 
    config = perform_resampling(config, path_table);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
elseif nargin>1 && isequal(step,'register')
    config = perform_registration(config,path_table_nii);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
elseif nargin>1 && isequal(step,'count')
    path_table = path_table(path_table.markers == config.markers(1),:);
    config = perform_counting(config,path_table);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
elseif nargin>1 && isequal(step,'classify')
    config = perform_classification(config, path_table);
    if nargout == 1; path_table = []; end
    if nargout < 1; clear config; end
    return
end

%% Run full analysis pipeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
config = perform_resampling(config, path_table);
config = perform_registration(config);
config = perform_counting(config,path_table);
config = perform_classification(config, path_table);
fprintf('%s\t Analysis steps completed! \n',datetime('now'))

end
