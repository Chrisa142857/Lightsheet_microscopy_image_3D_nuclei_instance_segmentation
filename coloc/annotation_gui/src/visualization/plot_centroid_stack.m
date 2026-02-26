function [L,z] = plot_centroid_stack(config, spacing, plot_classes, save_image)
%--------------------------------------------------------------------------
% Visualize image centroids at a corresponding z slice.
%--------------------------------------------------------------------------
% Usage:
% [L,z] = plot_centroid_stack(config, spacing, plot_classes, save_image)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analyis configuration structure. 
%
% spacing: (1x3 integer) Amount of downsampling for each dimension. 
% (default: [3,3,100])
% 
% plot_classes: (int,'none','all') Color by classificaiton if they exist.
% Specify cell class index. 'none': plot all cells as 1 class. 'all': plot
% all unique cell classes present. (default: 'all')
% 
% save_image: (logical) Save image to 'samples' directory. (default: true,
% if no output, false otherwise)
% 
%--------------------------------------------------------------------------
% Outputs:
% L: (uint8) 3D label image containing annotated centroid and class 
% coordinates.
%
% z: (int) z positions that were sampled.
%
%--------------------------------------------------------------------------
if nargin<2
    spacing = [3,3,100];
end
if nargin<3; plot_classes = 'all'; end
if nargin<4
    if nargout == 1
        save_image = false;
    else
        save_image = true;
    end
end

% Check for centroids
res_path = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));
if ~isfile(res_path)
    error("Could not locate results structure in %s output directory",config.sample_id)
else
    var_names = who('-file',res_path);
    if ismember('centroids',var_names)
        load(res_path,'centroids')
    else
        error("No centroids detected in results structure")
    end
    if ~isequal(plot_classes,'none')
        if ismember('classes',res_path)
            load(res_path,'classes')
            assert(length(classes) == size(centroids,1),...
                "Number of centroids and classes do not match")
        else
            warning("No classes detected in results structure")
            plot_classes = 'none';
        end
    end
end

% Get image paths
path_table = path_to_table(config);

% Get z positions after adjusting for spacing
z = path_table.z;
z = round(linspace(min(z),max(z),ceil(length(z)/spacing(3))+2));
z = z(2:end-1);
idx = ismember(centroids(:,3),z);
cen = centroids(idx,:);
if isequal(plot_classes,'none')
    classes = ones(1,size(cen,1));
else
    classes = classes(idx,:);
    if isnumeric(plot_classes)
        classes(~ismember(classes,plot_classes)) = 0;
    end
end

% Read a sample image to get dimensions
tempI = read_img(path_table);
dims = round(size(tempI)./spacing([1,2]));

% Create label image
L = zeros([dims,length(z)]);
se = strel('disk',5);
for i = 1:length(z)
    img = zeros(dims);
    idx = cen(:,3) == z(i);
    val = classes(idx);
    c = round(cen(idx,1:2)./spacing([1,2]));
    k_idx = all(c>1,2);
    k_idx = k_idx & c(:,1)<=dims(1) & c(:,2)<=dims(2);
    c = c(k_idx,:);
    val = val(k_idx);
    v_idx = sub2ind(dims,c(:,1),c(:,2));
    img(v_idx) = val;
    img = imdilate(img,se);
    L(:,:,i) = img;
end

% Save image
if save_image
    save_name = sprintf('%s_stack_labels.tif',config.sample_id);
    sample_dir = fullfile(config.output_directory,'samples');
    if ~isfolder(sample_dir)
        mkdir(sample_dir)
    end
    save_name = fullfile(sample_dir,save_name);
    fprintf('%s\t Writing image %s \n',datetime('now'),save_name)
    options.overwrite = true;
    options.compress = 'lzw';
    saveastiff(uint8(L),char(save_name),options);
end

if nargout<2
    clear z
end
if nargout<1
    clear L
end

end