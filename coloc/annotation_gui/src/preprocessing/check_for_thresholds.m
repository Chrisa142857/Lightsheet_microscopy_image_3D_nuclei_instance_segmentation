function config = check_for_thresholds(config,path_table,load_from_thresholds)
%--------------------------------------------------------------------------
% Check for intensity thresholds (i.e. thresholds.mat) structure and attach
% to config.
%--------------------------------------------------------------------------
% Usage: 
% config = check_for_thresholds(config,path_table)
% 
% Inputs:
% config - config structure from NM_process.
%
% path_table - table of file paths.
%
% Outputs:
% config = config structure with thresholds attached.
%--------------------------------------------------------------------------

if nargin<3
    load_from_thresholds = false;
end

var_file = fullfile(config.output_directory,'variables','thresholds.mat');

% First check if adj_params exists in variables folder
if isfile(var_file) || load_from_thresholds
    load(var_file,'thresholds')
    if all(thresholds.markers == config.markers) && all(thresholds.img_directory == config.img_directory)
        % Save into config
        config.lowerThresh = thresholds.lowerThresh;
        config.signalThresh = thresholds.signalThresh;
        config.upperThresh = thresholds.upperThresh;
        return
    end
end

% Check if thresholds already calculated and attached to adj_params
thresholds.img_directory = config.img_directory;
thresholds.markers = config.markers;

if ~isfield(config,'adj_params')
    fprintf("%s\t Lower and upper intensity thresholds are unspecified but are required "+...
        "for processing. Measuring these now... \n",datetime('now'));
    for i = 1:length(config.markers)
        [thresholds.lowerThresh(i), thresholds.upperThresh(i), thresholds.signalThresh(i)] = measure_images(config,path_table,i,true);
    end
    
    % Save into config
    config.lowerThresh = thresholds.lowerThresh;
    config.signalThresh = thresholds.signalThresh;
    config.upperThresh = thresholds.upperThresh;
else
    for i = 1:length(config.markers)
        thresholds.lowerThresh(i) = config.adj_params.(config.markers(i)).lowerThresh;
        thresholds.signalThresh(i) = config.adj_params.(config.markers(i)).signalThresh;
        thresholds.upperThresh(i) = config.adj_params.(config.markers(i)).upperThresh;
    end
end

% Save thresholds
save(var_file,'thresholds')

end