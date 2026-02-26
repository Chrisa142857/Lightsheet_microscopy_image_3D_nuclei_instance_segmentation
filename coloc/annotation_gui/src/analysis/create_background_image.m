function [B, S, centroids] = create_background_image(path_table_stitched, config, centroids)
% Create an image of local background intensities for cell counting

back_res = config.back_res; % Downsampled resolution of background images
res = config.resolution; % Voxel resolution
se_radius = 51; % Size of structuring element in um

% Make directory for storing background images
if exist(fullfile(config.output_directory,'background'),'dir') ~= 7
    mkdir(fullfile(config.output_directory,'background'))
end

% Since the background will be downsampled, we'll use 2D images to make
% processing quicker. Create a structuring element of defined size. This
% should be greater than the max radis of a cell nucleus
se = strel('disk',round(se_radius/res(1)));

B = cell(1,length(config.markers));
S = cell(1,length(config.markers));

for i = 2:length(config.markers)
    fprintf('%s\t Creating background image for marker %s \n',datetime('now'),...
        config.markers(i));
    
    % Read appropriate images based on specified resolutions
    path_table = path_table_stitched(path_table_stitched.markers == config.markers(i),:);
    tempI = imread(path_table.file{1});
    dims = size(tempI);
    res_dims = round([dims height(path_table)].*res./back_res);

    background = zeros(res_dims,'single');
    back_std = zeros(res_dims,'single');
    
    % Create a background image using a morphological opening filter with
    % a radius as defined above
    z_pos = round(linspace(1,height(path_table),res_dims(3)));
    smoothing = max(1,round(se_radius/10));
    parfor j = 1:length(z_pos)
        back_img = single(loadtiff(path_table.file{z_pos(j)}));
        back_img = imopen(back_img,se);
        back_img = imgaussfilt(back_img,2);
        background(:,:,j) = imresize(back_img,res_dims(1:2));
    end
    B{i} = background;
    
    % Create a standard deviation using a standard deviation filter to
    % measure the local variation of the background
    se2 = strel('sphere',round(se_radius./back_res(1)));
    back_std = single(stdfilt(background,se2.Neighborhood));
    S{i} = back_std;
    
    % Save images
    save_name = sprintf('%s_C%d_%s_background.nii',config.sample_name,...
    i,config.markers(i));
    save_location = fullfile(config.output_directory, 'background',save_name);
    niftiwrite(background,save_location)
    
    save_name = sprintf('%s_C%d_%s_back_std.nii',config.sample_name,...
    i,config.markers(i));
    save_location = fullfile(config.output_directory, 'background',save_name);
    niftiwrite(back_std,save_location)
    
    % Optional: if provided a centroid list, add background intensity
    % values to the matrix
    if nargin>2
        % Convert centroid positions to background image space
        cen_location = centroids(:,1:3).*res;
        cen_mask = round(cen_location./back_res);

        % Limit any indexes that fall out of range of the background image
        % after round
        cen_mask(any(cen_mask==0)) = 1;
        cen_mask(cen_mask(:,1) > res_dims(1)) = res_dims(1);
        cen_mask(cen_mask(:,2) > res_dims(2)) = res_dims(2);
        cen_mask(cen_mask(:,3) > res_dims(3)) = res_dims(3);

        % Get linear indexes for centroid positions in the background image
        cen_idx = sub2ind(res_dims, cen_mask(:,1),cen_mask(:,2), cen_mask(:,3));
        
        % Append values for background and standard deviation
        centroids = horzcat(centroids,single(B{i}(cen_idx)));
        centroids = horzcat(centroids,single(S{i}(cen_idx)));
    end
end

B{1} = zeros(size(B{2}));
S{1} = zeros(size(B{1}));

if nargin>2
    fprintf('%s\t Saving new centroid list \n',datetime('now'))
    path_centroids = fullfile(config.output_directory, sprintf('%s_centroids_updated.csv',config.sample_name));
    writematrix(centroids,path_centroids);
end

end

