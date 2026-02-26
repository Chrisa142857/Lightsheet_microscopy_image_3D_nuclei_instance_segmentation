function [path_table_series, path_table_nii] = path_to_table(config,location,quick_load,save_table)
%--------------------------------------------------------------------------
% Convert a structure of image directories, identifies tile positions and 
% marker names by regular expressions, and outputs a final table organized 
% with relevant position and channel information for futher processing. The
% file identifiers used here are specific to those used by the ImSpector 
% software used by LaVision Biotech. Images must be 2D, single channel
% series.
%--------------------------------------------------------------------------

% Check file location if not provided
if nargin<2
    if isstring(config) || ischar(config)
        % Directory location and not config structure
        fprintf("%s\t Reading image filename from single folder using default settings \n",datetime('now'))
        paths = string(config);
        assert(isfolder(paths) && length(paths) == 1,...
            "String input must be path specifying a single directory.")
        names = dir(paths);
        if any(arrayfun(@(s) contains(s.name,'aligned'),names))
            location = "aligned";
        elseif any(arrayfun(@(s) contains(s.name,'stitched'),names))
            location = "stitched";
        elseif all(arrayfun(@(s) contains(s.name,'.nii'),names)) && all(arrayfun(@(s) any(ismember(config.markers, s.name)),names))
            location = "resampled";
        else
            [path_table_series, path_table_nii] = munge_string(paths);
            return
        end    
    elseif isequal(config.use_processed_images,"aligned")
        % Start from after multi-channel alignment    
        fprintf("%s\t Reading image filename information from aligned directory \n",datetime('now'))
        location = "aligned";
    elseif isequal(config.use_processed_images,"stitched")
        % Start from after stitching
        fprintf("%s\t Reading image filename information from stitched directory \n",datetime('now'))
        location = "stitched";
    elseif isequal(config.use_processed_images,"resampled")
        % Start from after resampled
        fprintf("%s\t Reading image filename information from resampled directory \n",datetime('now'))
        location = "resampled";
    elseif endsWith(config.img_directory,'.csv')
        % Read from csv file
        fprintf("%s\t Reading image filename information from .csv file \n",datetime('now'))
        location = "csv";
    else
        %Start from raw image directory
        fprintf("%s\t Reading image filename information from raw image directory \n",datetime('now'))
        location = "raw";
    end
elseif isempty(location)
    %Start from raw image directory by default
    fprintf("%s\t Reading image filename information from raw image directory \n",datetime('now'))
    location = "raw";
end

% Quick load from saved path_table variable
if nargin<3
    quick_load = true;
end

% Overwrite saved path_table variable
if nargin<4
    save_table = false;
end

% Load table if variable exists
location = char(location);
path_table = [];

% For sshfs: if base directory specified, do not quick load from saved
% image paths. Ideally this would be updated to automatically use the base
% directory
if isstruct(config) && isfield(config,'base_directory') &&... 
~isempty(config.base_directory) && isfolder(config.base_directory)
    quick_load = false;
    save_table = false;
elseif isstruct(config) && isfile(fullfile(config.output_directory,'variables','path_table.mat'))
    var_location = fullfile(config.output_directory,'variables','path_table.mat');  
    load(var_location,'path_table')
    if ~isequal(location,"raw") && ~isfield(path_table,location) 
        quick_load = false;
        save_table = true;
    end
else
    save_table = true;
end

% Munge paths or read filename information from previously saved variable
switch location
    case 'raw'
        markers = config.markers;
        if isfield(config,'ignore_markers') && ~isempty(config.ignore_markers)
            markers = markers(~ismember(markers,config.ignore_markers));
            if isempty(markers)
                path_table_series = [];
                return
            end
        end
        if isempty(path_table) || ~quick_load
            [path_table_raw, path_table_nii] = munge_raw(config);
        else
            f = string(fieldnames(path_table));
            if ~all(ismember(markers,f))
                [path_table_raw, path_table_nii] = munge_raw(config);
                save_table = true;
            else
                fprintf("%s\t Quick loading from table \n",datetime('now'))
                for i = 1:length(markers)
                   path_table_raw.(markers(i)) = path_table.(markers(i)); 
                end
            end
        end 
        % Return if no tif but nii files present
        if isempty(path_table_raw) && ~isempty(path_table_nii)
            path_table_series = [];
            return
        end
        
    case 'aligned'
        if ~isempty(path_table) && quick_load
            fprintf("%s\t Quick loading from table \n",datetime('now'))
            path_table_series = path_table.aligned;
            return            
        end
        path_table_series = munge_aligned(config);
    case 'stitched'
        if ~isempty(path_table) && quick_load
            fprintf("%s\t Quick loading from table \n",datetime('now'))
            path_table_series = path_table.stitched;
            return
        end
        path_table_series = munge_stitched(config);
    case 'resampled'
        path_table_series = [];
        path_table_nii = munge_resampled(config);
        return
    case 'csv'
        if ~isempty(path_table) && quick_load
            path_table_series = path_table.raw;
            return
        end   
        path_table_series = readtable(config.img_directory);
    otherwise
        error("Unrecognized location selected")
end

% Save path_table for quicker loading next time
if save_table
    var_folder = fullfile(config.output_directory,'variables');  
    if ~isfolder(var_folder)
        mkdir(var_folder)
    end
    var_location = fullfile(var_folder,'path_table.mat');  
    
    fprintf("%s\t Saving path table \n",datetime('now'))
    if isequal(location,"raw")
        for i = 1:length(config.markers)
            path_table.(config.markers(i)) = path_table_raw.(config.markers(i)); 
        end
        if ~isempty(path_table_nii)
            path_table.nii = path_table_nii;
        end
    else
        path_table.(location) = path_table_series;
    end
    save(var_location,'path_table')
end

if isequal(location,"raw")  
    % Merge tables
    path_table_series = path_table_raw.(markers(1));
    z1 = unique(path_table_series.z)';
    y1 = unique(path_table_series.y)';
    x1 = unique(path_table_series.x)';
    for i = 2:length(markers)
        path_table_sub = path_table_raw.(markers(i));
        if ~all(x1 == unique(path_table_sub.x)')
            warning("Number of tile columns detected for marker %s does not "+...
                "match number of tile columns in reference channel.",markers(i))
        end
        if ~all(y1 == unique(path_table_sub.y)')
            warning("Number of tile rows detected for marker %s does not "+...
                "match number of tile columns in reference channel.",markers(i))
        end
        if ~all(z1 == unique(path_table_sub.z)')
            warning("Number of slices per tile detected for marker %s does not "+...
                "match number of tile columns in reference channel.",markers(i))
        end
        path_table_series = vertcat(path_table_series,path_table_sub);
    end
    
    % Check for duplicate files
    assert(length(unique(path_table_series.file)) == height(path_table_series),"Duplicate files added to multiple channels. "+...
        "Channel number indexes or unique folders for each marker are needed to import these images")
end

end
