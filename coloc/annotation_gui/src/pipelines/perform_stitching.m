function [config, path_table] = perform_stitching(config, path_table)
%--------------------------------------------------------------------------
% Run iterative 2D stitching.
%--------------------------------------------------------------------------
% Usage:
% [config, path_table] = perform_stitching(config, path_table)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Process configuration structure.
%
% path_table: Path table containing image information.
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
% path_table: Updated path table.
%
%--------------------------------------------------------------------------

% Check selection
if isequal(config.stitch_images,"false")
    fprintf("%s\t No image stitching selected. \n",datetime('now'));
    return
elseif ~isequal(config.stitch_images, "true") && ~isequal(config.stitch_images, "update")
    error("Unrecognized selection for stitch_images. "+...
    "Valid options are ""true"", ""update"", or ""false"".")
end

% Count number of x,y tiles
ncols = length(unique(path_table.x));
nrows = length(unique(path_table.y));
fprintf("%s\t Perfoming iterative 2D image stitching \n",datetime('now'));

% Check for aligned images
config.alignment_table = [];
if isequal(config.img_directory, fullfile(config.output_directory,'aligned'))
    fprintf("%s\t Using images from aligned directory \n",datetime('now'));
elseif isequal(config.load_alignment_params,"true") && isequal(config.save_images,"true") &&...
        ~all(config.stitch_sub_channel == 1)
    fprintf("%s\t Loading channel alignment translations \n",datetime('now'));
    if isfile(fullfile(config.output_directory,'variables','alignment_table.mat'))
        load(fullfile(config.output_directory,'variables','alignment_table.mat'),'alignment_table')
    else
        error("Could not locate alignment table")
    end

    % Check tiles
    empty_alignment_tiles = find(cellfun(@(s) isempty(s),alignment_table'));
    if empty_alignment_tiles
        error("No alignment parameters to apply for tiles %s",...
            sprintf("%s",num2str(empty_alignment_tiles)))
    end
    config.alignment_table = alignment_table;
end

% Check for intensity thresholds which are required for alignment
if isempty(config.lowerThresh) || isempty(config.upperThresh) || isempty(config.signalThresh)
    config = check_for_thresholds(config,path_table);
end

% Trim markers to only the ones specified
%if ~isempty(config.stitch_sub_channel)
%   markers = config.markers(config.stitch_sub_channel);
%   path_table = path_table(ismember(path_table.markers,markers),:);
%end

% Trim z positions that aren't present in all tiles
min_z_mat = zeros(nrows,ncols);
max_z_mat = zeros(nrows,ncols);
for i = 1:ncols
    for j = 1:nrows
        max_z_mat(j,i) = max(path_table.z(path_table.x==i & path_table.y==j));
        min_z_mat(j,i) = min(path_table.z(path_table.x==i & path_table.y==j));
    end
end
max_z = min(max_z_mat(:));
min_z = max(min_z_mat(:));
path_table(path_table.z>max_z,:)=[];
path_table(path_table.z<min_z,:)=[];

% Calculate adjustments in z
var_file = fullfile(config.output_directory,'variables','z_disp_matrix.mat');
if isequal(config.update_z_adjustment,"true") || ~isfile(var_file)
    % Calculate z displacements
    fprintf("%s\t Calculating z displacement matrix for stitching \n",datetime('now'));
    z_adj = calculate_adjusted_z(path_table,nrows,ncols,config.markers,...
            config.overlap,config.lowerThresh,config.output_directory);
    [~,z_idx] = setdiff(path_table.file,z_adj.file);
    path_table(z_idx,:) = [];
    path_table.z_adj = z_adj.z_adj;
else
    % Load z displacement info from a matrix
    fprintf("%s\t Loading z displacement matrix for stitching \n",datetime('now'));
    load(var_file, 'z_disp_matrix')
    assert(nrows == size(z_disp_matrix,1) && ncols == size(z_disp_matrix,2), "Loaded z adjustment parameters do not match number of tiles "+...
        "detected in the input image directory. Check input image directory or update z adjustment")
    path_table = apply_adjusted_z(path_table, z_disp_matrix);
end

% Check whether to stitch from previously calculated transforms.
update_stack = false;
if isequal(config.stitch_images,"update") && isempty(config.stitch_sub_stack)
    update_stack = true;
end

nb_images = length(min(path_table.z_adj):max(path_table.z_adj));
stitch_file = fullfile(config.output_directory,'variables','stitch_tforms.mat');
if isfile(stitch_file) && ~update_stack
    fprintf("%s\t Loading previously calculated stitching parameters \n",datetime('now'));
    load(stitch_file, 'h_stitch_tforms','v_stitch_tforms')
    loaded = true;

    % Check if sizes match up
    %if size(h_stitch_tforms,2) ~= nb_images
    %   error("%s\t Number of slices z slices in stitching parameters does not match loaded images \n",string(datetime('now')))
    if size(h_stitch_tforms,1) ~= (ncols-1)*nrows*2
       error("%s\t Number of column positions does not match loaded stitching parameters \n",string(datetime('now')))
    elseif size(v_stitch_tforms,1) ~= (nrows-1)*2
       error("%s\t Number of row positions does not match loaded stitching parameters \n",string(datetime('now')))
    end
else
    fprintf("%s\t Generating new stitching parameters \n",datetime('now'));
    h_pos = (ncols-1)*nrows*2;
    v_pos = (nrows-1)*2;
    h_stitch_tforms = zeros(h_pos,nb_images);
    v_stitch_tforms = zeros(v_pos,nb_images);
    save(stitch_file,'h_stitch_tforms','v_stitch_tforms','-v7.3');  
    update_stack = true;
    loaded = false;
end

if isequal(config.stitch_images,"true") && loaded
    % Check if all slices have been stitched. If not, switch to update
    % and stitch remaining slices
    if all(h_stitch_tforms==0,'all') && all(v_stitch_tforms==0,'all')
        update_stack = true;        
    elseif any(all(h_stitch_tforms == 0)) && any(all(v_stitch_tforms == 0))
        warning("Missing stitching transforms in loaded stitching parameters. "+...
            "Stitching will continue only for these slices. To re-stitch entire stack, "+...
            "set stitch_images to ""update"".");
        config.stitch_sub_stack = find(sum(h_stitch_tforms,1) == 0);
        start_z = find([diff(config.stitch_sub_stack)~=1,true],1);            
        if isempty(start_z)
            [~,idx] = min(abs(config.stitch_sub_stack - round(length(config.stitch_sub_stack)/2)));
            start_z = config.stitch_sub_stack(idx);
        end
        config.stitch_start_slice = start_z;
        update_stack = true;
    end
end

if isequal(config.stitch_images,"update") || update_stack
    % Perform stitching
    stitch_iterative(config,path_table)
else
    % Perform stitching using loaded parameters
    stitch_from_loaded_parameters(path_table, h_stitch_tforms, v_stitch_tforms, config)
end

fprintf("%s\t Stitching for sample %s completed! \n",datetime('now'),config.sample_id);

end
