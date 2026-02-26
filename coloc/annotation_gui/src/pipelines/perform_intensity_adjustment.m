function [config, path_table] = perform_intensity_adjustment(config, path_table, nrows, ncols)
%--------------------------------------------------------------------------
% Calculate image intensity adjustment parameters.
%--------------------------------------------------------------------------
% Usage:
% [config, path_table] = perform_intensity_adjustment(config, path_table, 
% nrows, ncols)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Process configuration structure.
%
% path_table: Path table containing image information.
%
% nrows: Number of tile rows in mosaic.
%
% ncols: Number of tile columns in mosaic.
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
% path_table: Updated path table.
%
%--------------------------------------------------------------------------

if length(config.adjust_tile_position) == 1
    config.adjust_tile_position = repmat(config.adjust_tile_position,1,length(config.markers));
end
if length(config.adjust_tile_shading) == 1
    config.adjust_tile_shading = repmat(config.adjust_tile_shading,1,length(config.markers));
end
if length(config.ls_width) == 1
    config.ls_width = repmat(config.ls_width,1,length(config.markers));
end
if length(config.laser_y_displacement) == 1
    config.laser_y_displacement = repmat(config.laser_y_displacement,1,length(config.markers));
end

% Check selection
if isequal(config.adjust_intensity,"false")
    % No intensity adjustments
    fprintf("%s\t No intensity adjustments selected \n",datetime('now'));
    config.adj_params = [];
    [~, config] = check_adj_parameters([],config);
    return
elseif ~isequal(config.adjust_intensity, "true") && ~isequal(config.adjust_intensity, "update")
    error("Unrecognized selection for adjust_intensity. "+...
    "Valid options are ""true"", ""update"", or ""false"".")
end

adj_file = fullfile(config.output_directory,"variables",'adj_params.mat');
loaded_params = false;

if isempty(config.update_intensity_channels) && isequal(config.adjust_intensity,"update")
    config.update_intensity_channels = 1:length(config.markers);
end

% Check for previously calculated parameters
if isfile(adj_file)
    fprintf("%s\t Loading adjustment parameters \n",datetime('now'));
    load(fullfile(config.output_directory,'variables','adj_params.mat'),'adj_params')

    % If all are loaded can return 
    if isequal(config.adjust_intensity, "true")
        [adj_params, config] = check_adj_parameters(adj_params,config,nrows,ncols);
        % Attach to config structure
        config.adj_params = adj_params;
        return  
    else
        loaded_params = true;
    end
end

if ~loaded_params || isequal(config.adjust_intensity,"update") &&...
        isempty(config.update_intensity_channels)
    % Create new adj_params structure for all channels
    config.update_intensity_channels = 1:length(config.markers);

    % Define intensity adjustment parameters from scratch. Intensity
    % adjustment measurements should be made on raw images.
    fprintf("%s\t Defining new adjustment parameters \n",datetime('now'));        
    adj_fields = {'adjust_tile_shading','adjust_tile_position',...
        'lowerThresh','upperThresh','signalThresh'};

    % Update parameters from config structure
    for i = 1:length(adj_fields)
        params.(adj_fields{i}) = config.(adj_fields{i});
    end
end

% Calculate intensity adjustments
 for k = 1:length(config.markers)            
    if ~ismember(k,config.update_intensity_channels)
        continue
    else
        marker = config.markers(k);
    end
    fprintf("%s\t Measuring intensity for %s \n",datetime('now'),config.markers(k));
    stack = path_table(path_table.markers == config.markers(k),:);
        
    % Save marker and img_location
    if length(config.img_directory) == length(config.markers)
        params.img_directory = config.img_directory(k);
    else
        params.img_directory = config.img_directory;
    end

    % Take measurements    
    [lowerThresh_measured, upperThresh_measured, signalThresh_measured, t_adj, y_adj, flatfield, darkfield] = ...
        measure_images(config, stack, k);

    % Save adjustments to parameter structure
    params.lowerThresh = lowerThresh_measured;
    params.upperThresh = upperThresh_measured;
    params.signalThresh = signalThresh_measured;
    
    params.y_adj = y_adj;
    params.t_adj = t_adj;
    params.flatfield = flatfield;
    params.darkfield = darkfield;
    params.darkfield_intensity = config.darkfield_intensity(k);
    
    % Save params to adj params structure
    adj_params.(marker) = params;
end

% Check for user-defined intensity threshold values
[adj_params, config] = check_adj_parameters(adj_params, config, nrows, ncols);
config.adj_params = adj_params;

% Save parameters and thresholds to output directory
fprintf("%s\t Saving adjustment parameters \n", datetime('now'));
save(fullfile(config.output_directory,'variables','adj_params.mat'), 'adj_params')

% Save flatfield and darkfield images as seperate variables
flat_file = fullfile(config.output_directory,'variables','flatfield.mat');
if isfile(flat_file)
    load(flat_file,'flatfield')
end
dark_file = fullfile(config.output_directory,'variables','darkfield.mat');
if isfile(dark_file)
    load(dark_file,'darkfield')
end
clear flatfield; clear darkfield
save_flat = false;
for k = 1:length(config.markers)
    flat = adj_params.(marker).flatfield;
    dark = adj_params.(marker).darkfield;
    if all(flat(:) ~= 1)
        flatfield.(marker) = flat;
        darkfield.(marker) = dark;
        save_flat = true;
    end
end
if save_flat
    save(flat_file, 'flatfield')
    save(dark_file, 'darkfield')
end

config = check_for_thresholds(config,path_table);
config.adjust_intensity = "true";

% Save flatfield and darkfield as seperate variables
if isequal(config.adjust_tile_shading,"basic")
    fprintf("%s\t Saving flatfield and darkfield images \n",datetime('now'));
    save(fullfile(config.output_directory,'variables','flatfield.mat'), 'flatfield')
    save(fullfile(config.output_directory,'variables','darkfield.mat'), 'darkfield')
end

% Save samples
if isequal(config.save_samples,"true")
    fprintf('%s\t Saving samples \n',datetime('now'));    
    save_samples(config,'intensity',path_table)
end

end
