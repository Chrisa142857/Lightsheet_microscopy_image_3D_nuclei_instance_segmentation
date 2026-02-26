function [L,I] = plot_centroid_patch(config, coordinates, dims, plot_classes, save_image)
%--------------------------------------------------------------------------
% Extract boxed region around centroid coordinates.
%--------------------------------------------------------------------------
% Usage:
% [L,I] = plot_centroid_patch(config, coordinates, dims, plot_classes, save_image)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analyis configuration structure. 
%
% cooridnates: (1x3 numeric) z,y,x coordinates at the center of boxed
% region.
%
% dims: (1x3 int) z,y,x dimensions of boxed region.
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
% B: (uint8) 3D label image containing annotated centroid and class 
% coordinates.
%
%--------------------------------------------------------------------------
if nargin<2
    error("Please specify centroid coordinates for the patch region")
end

if nargin<3    
    dims = [60,200,200];
end
if nargin<4; plot_classes = 'all'; end
if nargin<5
    if nargout > 0
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

% Get patch ranges
c = round(coordinates);
slices = c(1) - floor(dims(1)/2):c(1) + ceil(dims(1)/2);
rows = c(2) - floor(dims(2)/2):c(2) + ceil(dims(2)/2);
cols = c(3) - floor(dims(3)/2):c(3) + ceil(dims(3)/2);

if min(slices)<1 || max(slices)>max(path_table.z)
    error("Selected z positions fall outside of image range")
end

% Read a sample image to get dimensions
dims = size(read_img(path_table));
if min(rows)<1 || max(rows)>dims(1)
    error("Selected y positions fall outside of image range")
end

if min(cols)<1 || max(cols)>dims(2)
    error("Selected x positions fall outside of image range")
end

% Get z positions after adjusting for spacing
z = min(slices)-1:max(slices)+1;
idx = ismember(centroids(:,1),rows);
idx = idx & ismember(centroids(:,2),cols);
idx = idx & ismember(centroids(:,3),z);
cen = centroids(idx,:);

% Print total number of centroids detected
fprintf("Total centroid count in patch: %d\n",size(cen,1))

if isequal(plot_classes,'none')
    classes = ones(1,size(cen,1));
else
    classes = classes(idx,:);
    if isnumeric(plot_classes)
        classes(~ismember(classes,plot_classes)) = 0;
    end
end

% Read images
I = zeros(length(rows),length(cols),length(slices),'uint16');
for i = 1:length(slices)
    I(:,:,i) = read_img(path_table,[1,slices(i)],[rows(1),rows(end),cols(1),cols(end)]);
end

% Create label image
L = zeros([dims,length(z)-2],'uint8');
se = strel([0 1 0;1 1 1;0 1 0]);
for i = 2:length(z)-1
    for j = -1:1
        img = zeros(dims,'uint8');
        idx = cen(:,3) == (z(i) + j);
        val = classes(idx);
        c1 = cen(idx,1:2);
        k_idx = all(c1>1,2);
        k_idx = k_idx & c1(:,1)<=dims(1) & c1(:,2)<=dims(2);
        c1 = c1(k_idx,:);
        val = val(k_idx);
        v_idx = sub2ind(dims,c1(:,1),c1(:,2));
        img(v_idx) = val;
        if j == 0
            img = imdilate(img,se);
        end
        L(:,:,i-1) = L(:,:,i-1) + img;
    end
end
L = L(rows,cols,1:length(slices));

% Save image
if save_image
    sample_dir = fullfile(config.output_directory,'samples');
    if ~isfolder(sample_dir)
        mkdir(sample_dir)
    end
    options.overwrite = true;
    options.compress = 'lzw';
    save_name = sprintf('%s_patch_%d-%d_%d-%d_%d-%d.tif',config.sample_id,...
        slices(1),slices(end),rows(1),rows(end),cols(1),cols(end));
    save_name = fullfile(sample_dir,save_name);
    fprintf('%s\t Writing image %s \n',datetime('now'),save_name)
    saveastiff(im2uint8(imadjustn(I,[min(double(I(:)))/65535,max(double(I(:)))/65535])),...
        char(save_name),options);

    save_name = sprintf('%s_labels_%d-%d_%d-%d_%d-%d.tif',config.sample_id,...
        slices(1),slices(end),rows(1),rows(end),cols(1),cols(end));
    save_name = fullfile(sample_dir,save_name);
    saveastiff(im2uint8(imadjustn(I,[min(double(I(:)))/65535,max(double(I(:)))/65535])),...
        char(save_name),options);
    saveastiff(uint8(L),char(save_name),options);
end

if nargout<2
    clear I
end
if nargout<1
    clear L
end

end