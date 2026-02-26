function t_adj = adjust_tile_multi(config, path_table, defaults)
%--------------------------------------------------------------------------
% Calculate intensity adjustment for multi-tile layout by measuring
% overlapping regions.
%--------------------------------------------------------------------------
%
% Inputs:
% config - config structure from NM_process.
%
% path_table - path table containing images to measure. Should only contain
% a single channel.
%
% defaults - structure containing intensity percentiles, padding, and image
% sampling. See measure_images.
%
% Outputs:
% adj_matrix1 - adjustment matrix containing lower intensity thresholds.
%
% adj_matrix2 - adjustment matrix containing upper intensity thresholds. 
%--------------------------------------------------------------------------


% Load defaults
low_prct = defaults.low_prct;   % Low percentile for sampling background pixels
high_prct = defaults.high_prct; % High percentile for sampling bright pixels
pads = defaults.pads;           % Crop this fraction of image from the sides of images
image_sampling = defaults.image_sampling;   % Fraction of all images to sample

overlap = config.overlap;

% Count number of images and measure image dimensions
x_tiles = length(unique(path_table.x));
y_tiles = length(unique(path_table.y));
nb_slices = height(path_table)/(x_tiles*y_tiles);

% Get image positions
s = round(image_sampling*nb_slices);
img_range = round(linspace(min(path_table.z),max(path_table.z),s));

% Read image size and get padding
tempI = imread(path_table.file{1});
[nrows, ncols] = size(tempI);
pad_h = round(pads*ncols*overlap);
pad_v = round(pads*nrows*overlap);

% Determine overlap region
% Horizontal image overlap regions
overlap_min_c = 1:round(ncols*overlap)-pad_h;
overlap_max_c = ncols-round(ncols*overlap)+1:ncols-pad_h;
overlap_min_h = {[1,nrows],[1,overlap_min_c(end)]};
overlap_max_h = {[1,nrows],[overlap_max_c(1),overlap_max_c(end)]};

% Vertical image overlap regions
overlap_min_r = 1:round(nrows*overlap)-pad_v;
overlap_max_r = nrows-round(nrows*overlap)+1:nrows-pad_v;
overlap_min_v = {[1,overlap_min_r(end)],[1,ncols]};
overlap_max_v = {[overlap_max_r(1),overlap_max_r(end)],[1,ncols]};

% Measure pairwise horizontal intensity differences
fprintf('%s\t Measuring between tile differences horizontally \n',datetime('now'));    
h_matrix = ones(y_tiles,x_tiles);
for i = 1:y_tiles
    for j = 1:x_tiles-1
        I_left = zeros([nrows round(ncols*overlap)-pad_h length(img_range)],'uint16');
        I_right = zeros([nrows round(ncols*overlap)-pad_h length(img_range)],'uint16');
        
        for k = 1:length(img_range)
            % Read image regions where tiles should overlap
            file_left = path_table(path_table.y == i & path_table.x == j & path_table.z == img_range(k),:);
            file_right = path_table(path_table.y == i & path_table.x == j+1 & path_table.z == img_range(k),:);
            
            I_left(:,:,k) = imread(file_left.file{1},'PixelRegion',overlap_max_h);
            I_right(:,:,k) = imread(file_right.file{1},'PixelRegion',overlap_min_h);
        end

        % Remove any zero values from shifting image
        I_left = single(I_left(I_left>0));
        I_right = single(I_right(I_right>0));

        % Apply flatfield + darkfield corrections
        %I_left = (single(I_left)-d_left)./f_left + d_left;
        %I_right = (single(I_right)-d_right)./f_right + d_right;

        % Difference in average intensity indicates between tile differences
        I_left = I_left*h_matrix(i,j);
        I_right = I_right*h_matrix(i,j+1);
        h_matrix(i,j+1) = prctile(I_left,high_prct)/prctile(I_right,high_prct);
    end
end
h_matrix = h_matrix/mean2(h_matrix);

% Measure pairwise vertical intensity differences
fprintf('%s\t Measuring between tile differences vertically \n',datetime('now'));    
v_matrix = ones(y_tiles,x_tiles);
for i = 1:y_tiles-1
    for j = 1:x_tiles
        I_top = zeros([round(nrows*overlap)-pad_v ncols length(img_range)],'uint16');
        I_bottom = zeros([round(nrows*overlap)-pad_v ncols length(img_range)],'uint16');
        for k = 1:length(img_range)
            % Read image regions where tiles should overlap
            file_top = path_table(path_table.y == i & path_table.x == j & path_table.z == img_range(k),:);
            file_bottom = path_table(path_table.y == i+1 & path_table.x == j & path_table.z == img_range(k),:);

            I_top(:,:,k) = imread(file_top.file{1},'PixelRegion',overlap_max_v);
            I_bottom(:,:,k) = imread(file_bottom.file{1},'PixelRegion',overlap_min_v);  
        end

        % Remove any zero values from shifting image
        I_top = single(I_top(I_top>0));
        I_bottom = single(I_bottom(I_bottom>0));

        % Apply flatfield + darkfield corrections
        %I_top = (single(I_top)-d_top)./f_top + d_top;
        %I_bottom = (single(I_bottom)-d_bottom)./f_bottom + d_bottom;

        %Difference in average intensity indicates between tile differences
        I_top1 = I_top*v_matrix(i,j);
        I_bottom1 = I_bottom*v_matrix(i+1,j);
        v_matrix(i+1,j) = prctile(I_top1,high_prct)/prctile(I_bottom1,high_prct);
    end
end
v_matrix = v_matrix/mean2(v_matrix);

% Combine matrices by row-wise multiplicaiton of each row
adj_matrix1 = zeros(size(h_matrix));
for i = 1:size(h_matrix,1)
    adj_matrix1 = adj_matrix1 + v_matrix.*h_matrix(i,:)/size(h_matrix,1); 
end

% Combine matrices by column-wise multiplication of column
adj_matrix2 = zeros(size(v_matrix));
for i = 1:size(v_matrix,2)
    adj_matrix2 = adj_matrix2 + v_matrix(:,i).*h_matrix/size(h_matrix,2);
end

% Calculate final adjustments by averaging column-wise/row-wise matrices
t_adj = (adj_matrix1+adj_matrix2)/2;

% Display final adjustment matrices
fprintf('%s\t Final Tile Adjustment Matrices:\n',datetime('now'));    
disp(t_adj)

end