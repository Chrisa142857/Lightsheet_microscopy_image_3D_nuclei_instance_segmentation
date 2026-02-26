function [L,I] = plot_centroid_slice(config, z_position, plot_classes, save_image)
%--------------------------------------------------------------------------
% Visualize image centroids on a corresponding z slice.
%--------------------------------------------------------------------------
% Usage:
% I_final = plot_centroid_slice(config, z_position, plot_classes, 
% save_image)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analyis configuration structure. 
%
% z_position: (integer) z position to visualize.
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
% L: (uint8) 2D label image containing annotated centroid coordinates.
%
% I: (uint16) 2D target image.
%
%--------------------------------------------------------------------------
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
path_table = path_table(path_table.channel_num == 1,:);

% Get z position at current slice and slices above/below
z_position = z_position-1:z_position+1;
idx = ismember(centroids(:,3),z_position);
cen = centroids(idx,:);
if isequal(plot_classes,'none')
    classes = ones(1,size(cen,1));
else
    classes = classes(idx,:);
    if isnumeric(plot_classes)
        classes(~ismember(classes,plot_classes)) = 0;
    end
end
I = read_img(path_table,[1,z_position(2)]);

% Create label image
L = zeros(size(I),'uint8');
se = strel([0 1 0;1 1 1;0 1 0]);
for i = 1:3
    img = zeros(size(I),'uint8');
    idx = cen(:,3) == z_position(i);
    val = classes(idx);
    idx = sub2ind(size(I),cen(idx,1),cen(idx,2));
    img(idx) = val;
    if i == 2
        img = imdilate(img,se);
    end
    L = L + img;
end

% Save image
if save_image
    I = imadjust(I);
    save_img = labeloverlay(I,L,'ColorMap','prism');
    save_name = sprintf('%s_slice_%d.png',config.sample_id,z_position(2));
    sample_dir = fullfile(config.output_directory,'samples');
    if ~isfolder(sample_dir)
        mkdir(sample_dir)
    end
    save_name = fullfile(sample_dir,save_name);
    fprintf('%s\t Writing image %s \n',datetime('now'),save_name)
    imwrite(save_img,save_name)
end

if nargout<2
    clear I
end
if nargout<1
    clear L
end

end