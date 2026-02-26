function [lowerThresh, upperThresh, signalThresh, t_adj, y_adj, flatfield, darkfield] = measure_images(config, stack, channel_idx, get_thresholds)
%--------------------------------------------------------------------------
% Calculate various intensity adjustments including tile differences,
% light-sheet width correction, flatfield + darkfield shading correction
%--------------------------------------------------------------------------
% Some additional defaults for tile position adjustment
defaults.low_prct = 5;   % Low percentile for sampling background pixels
defaults.high_prct = 98; % High percentile for sampling bright pixels
defaults.pads = 0.05;     % Crop this fraction of image from the sides of images. Note this should be <0.5
defaults.image_sampling = 0.05;   % Fraction of all images to sample

% If just measuring thresholds, set to true
if nargin<4
    get_thresholds = false;
end

% Load config parameters
resolution = config.resolution{channel_idx};

% Make sure only selected channel is chosen
stack = stack(stack.markers == config.markers(channel_idx),:);
if isempty(stack)
    error("No images found in path table for marker %s",config.markers(channel_idx));
end

% Count number of images and measure image dimensions
x_tiles = length(unique(stack.x));
y_tiles = length(unique(stack.y));

% Read image size
tempI = imread(stack.file{1});
[nrows, ncols] = size(tempI);

% Get intensity thresholds
if isempty(config.lowerThresh) || isempty(config.upperThresh) || isempty(config.signalThresh)
    [lowerThresh, upperThresh, signalThresh] = measure_thresholds(stack,defaults);
else
    lowerThresh = config.lowerThresh(channel_idx); 
    upperThresh = config.upperThresh(channel_idx); 
    signalThresh = config.signalThresh(channel_idx);
end

% If just getting thresholds, measure those and return
if get_thresholds
    return
end

% Calculate and adjust for laser width
if isequal(config.adjust_tile_shading(channel_idx),"manual")
    y_adj = adjust_ls_width_measured(config.single_sheet(channel_idx),tempI,...
    config.ls_width(channel_idx),resolution,config.laser_y_displacement(channel_idx))';
else
    y_adj = ones(nrows,1);
end  

% Calculate shading correction using BaSiC
if isequal(config.adjust_tile_shading(channel_idx),"basic")
    [flatfield, darkfield] = estimate_flatfield(config, stack);
    % Check value in flatfield image
    if any(flatfield(:)>1.5) || any(flatfield(:)<0.5)
        warning("BaSiC measured some relatively large adjustment values in the "+...
            "flatfield image which may not be accurate. Try subsetting to tile positions with uniform signal "+...
            "throughout the full field of view. Otherwise BaSiC may not be the appropriate "+...
            "method for shading correction.")
        pause(5)
    end
    flatfield = single(flatfield);
    darkfield = single(darkfield);
else
    flatfield = ones(nrows,ncols,'single'); darkfield = ones(nrows,ncols,'single');
end

% Tile intensity adjustments
if x_tiles*y_tiles>1 && isequal(config.adjust_tile_position(channel_idx),"true")
    % Measure overlapping tiles and get thresholds
    defaults.image_sampling = min(defaults.image_sampling*10,1);
    t_adj = adjust_tile_multi(config,stack,defaults);
else
    if x_tiles*y_tiles == 1
        fprintf('%s\t Only 1 tile detected for marker %s \n',...
            datetime('now'),config.markers(channel_idx));
    end
    % Store default tile adjustments
    t_adj = [];
end

end
