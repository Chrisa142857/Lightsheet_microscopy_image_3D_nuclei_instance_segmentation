function resampled_I = resample_image(path_table, resolution, resample_resolution, channels)
%--------------------------------------------------------------------------
% Resample mage to specified resolution. Only requires NM_analysis config 
% structure, desired resolution, and image filetype. One file contains
% entire stack.
%--------------------------------------------------------------------------
%
% Usage:
% resampled_I = resample_image(path_table, resolution, resample_resolution, channels)
%
% Inputs:
% path_table - table of image file paths from path_to_table.
% 
% resolution - (1x3 numeric) input image resolution. 
%
% resample_resolution - (1x3 numeric) target resampling resolution.
%
% channels - (default: all channels) subset specific channel number to
% resample.
%
% Outputs:
% resampled_I - resampled image.
%--------------------------------------------------------------------------

% Check if path_table is string
if ischar(path_table) || isstring(path_table)
    path_table = path_to_table(path_table);
end

if length(resolution) == 1
    resolution = repmat(resolution,1,3);
end

if length(resample_resolution) == 1
    resample_resolution = repmat(resample_resolution,1,3);
end

markers = unique(path_table.markers); 
if nargin==4
    a = unique([path_table.markers,path_table.channel_num],'rows');
    markers = a(ismember(str2double(a(:,2)),channels));
    [~,idx] = sort(str2double(a(:,2)));
    markers = markers(idx);
end

resampled_I = cell(1,length(markers));
for i = 1:length(markers)
    % Subset marker
    path_sub = path_table(path_table.markers == markers(i),:);
    nb_images = height(path_sub);
    
    % Measure image dimensions for high resolution image
    tempI = imread(path_sub.file{1});
    [img_height,img_width] = size(tempI);

    % Calculate image dimensions for target resolution
    re_v = resample_resolution./resolution;
    re_height = round(img_height/re_v(1));
    re_width = round(img_width/re_v(2));
    re_z = round(nb_images/re_v(3));

    % Intialize matrix for resampled image
    re_I = zeros(re_height,re_width,re_z,'uint16');

    % Resample first only in x,y to decrease memory demands
    fprintf('%s\t Resampling in xy \n',datetime('now'))
    for j = 1:nb_images
        I = loadtiff(path_sub.file{j});
        re_I(:,:,j) = imresize(I,[re_height,re_width]);
    end

    % Then resample in z 
    fprintf('%s\t Resampling in z \n',datetime('now'))
    re_I = imresize3(re_I,[re_height,re_width,re_z]);
    resampled_I{i} = uint16(re_I);
end

if length(resampled_I) == 1
    resampled_I = resampled_I{1};
end

end