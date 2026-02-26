function [patches, ftable, cen_sub, cen_idx] = get_centroid_patches(centroids, path_table, config)
%--------------------------------------------------------------------------
% Get 2D image pacthes around centroid positions
%--------------------------------------------------------------------------
% Inputs:
% centroids - matrix with centroids positions + mask annotations.
%
% path_table_stitched - input image table.
%
% config - config structure from NM_analyze.
%
% patch_size - (optional) 1x2 element vector. 1st element is 0.5xlength of
% saved patch image for viewing. 2nd element is 0.5xlength of the actual 
% sampling window. Should be less than 1st element and ideally set to the
% approximate radius of a nucleus. 
%
% k - (optional) number of patches to generate (default: 1000 patches)
%
% low_thresh - (optional) threshold for minimum intensity for all channels
% set as fraction (i.e. 0-1). For example, if set to 0.5, nuclei with
% intensities below the 50th percentile in all channels will be presumed
% negative and ignored from patch generation.
%
% Outputs:
%
%
%--------------------------------------------------------------------------

% Defaults
is_cortex = false;

% Get params from config structure
patch_size = config.patch_size(1);                      
class_size = config.patch_size(2);
k = config.n_patches;
low_thresh = config.min_class_thresh;
nuc_channel = config.contains_nuclear_channel;

fprintf('\t Using sampling window of %d pixels\n',patch_size)
fprintf('\t Using classification window of %d pixels\n',class_size)

% Make an output directory
save_directory = fullfile(config.output_directory,'classifier');
if ~isfolder(save_directory)
    mkdir(save_directory);
end

% Can only do 3 markers currently
markers = config.markers;
if length(markers)>3
    markers = markers(1:3);
end

% Check for nuclear channel
if isequal(nuc_channel,"true")
    a = 1;
else
    a = 0;
end

% Check for minimum threshold
if isempty(low_thresh)
    low_thresh = 0;
end

if ~isequal(centroids,'load')
    % Merge centroid centroids structure to matrix
    % fprintf(centroids)
    centroids = [centroids.coordinates, centroids.annotations, centroids.intensities];

    % Subset cells above minimum intensity threshold
    k_idx = zeros(size(centroids,1),1);
    for i = 1+a:length(markers)
       idx = i + 4;
       thresh = prctile(centroids(:,idx),low_thresh*100);
       k_idx = k_idx | centroids(:,idx)>=thresh;
    end
    fprintf('\t Retaining %d from %d that are above the threshold %f \n',...
        sum(k_idx),length(k_idx),low_thresh)
    cen_sub = centroids(k_idx,:);
    cen_idx = find(k_idx)';
    
    % Shuffle indexes
    s_idx = randperm(size(cen_sub,1));
    cen_idx = cen_idx(s_idx);
    cen_sub = cen_sub(s_idx,:);
    
    fprintf('\t Selecting %d random patches \n',k)
    
    %cen_sub = zeros(k,3);
    %cen_idx = zeros(1,k);

    % Take random sample from indexes
    %s_idx = randsample(size(cen_sub,1),k);
    %fprintf('\t Selecting %d random patches \n',k)

    %cen_sub = cen_sub(s_idx,:);
    %cen_idx = cen_idx(s_idx);
    
    %[~,i] = sort(cen_sub(:,3));
    %cen_sub = cen_sub(i,:);
    %cen_idx = cen_idx(i)';
else
   % Load previous patch info list
   fprintf('\t Loading previous patch info \n')
   patch_name = fullfile(save_directory,sprintf('%s_patch_info.csv',config.sample_id));
   patch_info = readmatrix(patch_name); 
   cen_idx = patch_info(:,1)';
   cen_sub = patch_info(:,2:end);
end

patches = zeros([2*patch_size+1,2*patch_size+1,k],'uint16');
patches = repmat({patches},1,3);
ftable = cell(size(cen_sub,1),length(markers));
cen_idx_save = false(length(cen_idx),1);
a = 1;
b = 1;
while a<k+1
    % Get positions
    i = cen_idx(b);
    z = cen_sub(b,3);
    pos = cen_sub(b,1:2);
    
    % Get file
    file = path_table(path_table.z == z,1).file;
    ranges = {[pos(1)-patch_size,pos(1)+patch_size], [pos(2)-patch_size,pos(2)+patch_size]};
    
    c = 0;
    for j = 1:length(markers)
        % Read patch
        img = imread(file{j},'PixelRegion',ranges);
        
        if ~isequal(size(img),repmat(patch_size*2+1,1,2))
            break
        end

        % Get features from patch
        ftable{a,j} = measure_patch_features(img, class_size, true, config.markers(j));
        
        % Add to stack
        patches{j}(:,:,a) = img;
        c = 1;
    end
    if c == 1
        cen_idx_save(b) = 1;
        a = a+1;
    end
    b = b+1;
end
cen_idx = cen_idx(cen_idx_save);
cen_sub = cen_sub(cen_idx_save,:);

% Adjust intensity
for i = 1:length(markers)
    min_int = prctile(double(patches{i}(:)),0.02)/65535;
    max_int = prctile(double(patches{i}(:)),99.98)/65535;
    in_int = [min_int, max_int];
    patches{i} = im2uint8(imadjustn(patches{i}, in_int));
end

patches = cat(4,patches{:});
patches = permute(patches,[1,2,4,3]);

ftable1 = cell(1,length(markers));
for i = 1:length(markers)
   ftable1{i} = cat(1,ftable{:,i});
end
ftable = cat(2,ftable1{:});

% Save patch
options.overwrite = true;
options.color = true;
options.message = false;
img_name = fullfile(save_directory,sprintf('%s_patches.tif',config.sample_id));
saveastiff(patches,char(img_name),options);

% Save patch info
patch_info = horzcat(cen_idx',cen_sub);
patch_name = fullfile(save_directory,sprintf('%s_patch_info.csv',config.sample_id));
writematrix(patch_info,patch_name)

% Save patch features
% Get layer and structure info

% For cortex
if is_cortex
    structures = string(bin_annotation_structures(cen_sub(:,4),'cortex'));
    layers = string(bin_annotation_structures(cen_sub(:,4),'layers'));
    feature_table = array2table([layers,structures],'VariableNames',{'Layer', 'Structure'});
    feature_table = horzcat(feature_table,ftable);
else
    feature_table = ftable;
end

table_name = fullfile(save_directory,sprintf('%s_patch_features.csv',config.sample_id));
writetable(feature_table,table_name)

end