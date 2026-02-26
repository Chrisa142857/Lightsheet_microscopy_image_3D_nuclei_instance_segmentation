function config = perform_resampling(config, path_table)
%--------------------------------------------------------------------------
% Generate downsamppled images for registration.
%--------------------------------------------------------------------------
% Usage:
% config = perform_resampling(config, path_table)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analysis configuration structure.
%
% path_table: Path table containing image information.
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
%--------------------------------------------------------------------------

% Read parameters
res = config.resample_resolution;
res_path = fullfile(config.output_directory,'resampled');
config.resampled_paths = cell(1,length(config.markers));
flags = true(1,length(config.markers));

% Get channels to resample
if isempty(config.resample_channels)
    if ~isempty(config.registration_channels)
        resample_channels = config.registration_channels;
    else
        resample_channels = unique(path_table.channel_num)';
    end
    flags(~resample_channels) = false;
else
    resample_channels = config.resample_channels;
end

% Create resampled directory
if ~isfolder(res_path)
    mkdir(res_path)
else
    if ~isequal(config.resample_images,"update")
        % Check for images alread resampled
        for i = resample_channels
            filename = fullfile(res_path,sprintf('%s_C%d_%s_%d.nii',...
                config.sample_id,i,config.markers(i),res));
            if ~isfile(filename)
                flags(i) = false;
            end
        end
        
        % If all channels resampled at correct resolution, return
        if ~isequal(config.resample_images,"update")
            if all(flags)
                fprintf('%s\t All selected channels have been resampled\n',datetime('now'))            
                return
            end
        end
    end
end

% Perform resampling
resample_path_table(path_table, config, resample_channels);

end