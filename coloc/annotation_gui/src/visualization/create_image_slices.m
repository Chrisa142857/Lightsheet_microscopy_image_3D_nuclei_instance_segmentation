function stacks = create_image_slices(config, class_idx, spacing, disk_r)
%--------------------------------------------------------------------------
% Create centroid overlay for full image stack. This function is to get a
% downsampled overlay for the entire image dataset. Data is read from
% results structure. 
%--------------------------------------------------------------------------
% Usage:
% stack = create_image_slices(config, classes, resampling, spacing)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Configuration structure from analysis stage.
%
% class_idx: Classes to display. For displaying only centroids or if 
% classes haven't been calculated, specify this as 'nuclei'. Leaving empty 
% will generate stacks for all classes. Otherwise, specify specific class
% indexes. (default: [])
%
% spacing: (1x3 integer) Amount of downsampling for each dimension in each
% stack (e.g.: [10,10,100]). (default: 10% of each axis dimension)
%
% disk_r: (int) Radius of disk of element to mark centroids. (default: 1)
%--------------------------------------------------------------------------
% Outputs:
% stacks: Cell array of output stacks (optional). If no output specified, 
% image stacks will be saved to config output folder. If output is specified, image 
% stacks are not saved.
%
%--------------------------------------------------------------------------

% Defaults
if nargin < 2
    class_idx = [];
end

if nargin < 3
    spacing = [];
end

if nargin < 4
    disk_r = 1;
end

% Resolve whether to save
save_flag = true;
if nargout == 1
    save_flag = false;
else
    save_directory = fullfile(config.output_directory, 'samples','stacks');
    if ~isfolder(save_directory)
        mkdir(save_directory)
    end
    fprintf("Saving image stacks to %s\n", save_directory);
end

% Check config stage
if ~isequal(config.stage,"analyze")
    error("Analysis configuration structure required")
end

% Get path_table
path_table = path_to_table(config);
path_table.file{1} = '/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/data/pair3/output_L35D719P4/stitched/L35D719P4_0001_C1_topro_stitched.tif';

% Get dimensions
tempI = imread(path_table.file{1});
dims = [size(tempI),length(unique(path_table.z))];

% Rescale dims
if isempty(spacing)
    new_dims = round(dims*0.1);
    spacing = round(dims./new_dims);
    dims = new_dims;
else
    dims = round(dims./spacing);
end

% Query results structure
results_path = fullfile(config.output_directory, strcat(config.sample_id, '_results.mat'));
if ~isfile(results_path)
    error("No results structure in the output directory")
else
    results = load(results_path);
    if ~isfield(results, 'centroids')
        error("No centroid information was found in the results structure. Re-run cell detection")
    else
        centroids = results.centroids(:,1:3);
        rm_idx = false(size(centroids,1),1);
    end

    class_names = "nuclei";
    if ~isfield(results, 'classes')
        warning("No cell-type classifications were found in the results structure. Labeling only nuclei")
        classes = ones(1,size(centroids,1));
        classes(1:10000) = 2;
    else
        classes = results.classes;
        if isequal(class_idx, 'nuclei')
            classes = ones(1,size(centroids,1));
        elseif ~isempty(classes)
            rm_idx = rm_idx | ismember(classes, class_idx);
            class_names = config.markers(class_idx);
        end
        if isempty(class_idx)
            class_names = config.markers;
        end
    end
end

% Get z positions
z_positions = floor(linspace(min(path_table.z), max(path_table.z), dims(3)));

% Remove centroids outside of bounds
centroids(:,1:2) = round(centroids(:,1:2)./spacing(1:2));
for i = 1:2
    rm_idx = rm_idx | (centroids(:,i) < 1);
    rm_idx = rm_idx | (centroids(:,i) > dims(i));
end
rm_idx = rm_idx | ~ismember(centroids(:,3),z_positions);
centroids(:,3) = floor(centroids(:,3)./spacing(3));

% Subset centroids and classes
centroids = centroids(~rm_idx,:);
classes = classes(~rm_idx);
se = strel('disk',disk_r);

% Generate stacks
stacks = cell(1,length(unique(classes)));

for i = 1:length(unique(classes))
    stack = zeros(dims, 'logical');
    sub = centroids(classes == i, :);
    idx = sub2ind(dims, sub(:,1), sub(:,2), sub(:,3));
    stack(idx) = true;
    stack = imdilate(stack,se);

    % Save stacks to output folder
    if save_flag
        filepath = fullfile(save_directory, sprintf("%s_%s_%d_stack.tif",...
            config.sample_id, class_names(i), dims(1)));
        options.overwrite = true;
        options.verbose = true;
        options.compress = 'lzw';
        saveastiff(uint8(stack*255), char(filepath), options);
    else
        stacks{i} = uint8(stack*255);
    end
end

% Merge to array
stacks = cat(4,stacks{:});

end