
function [global_z_img_pos] = min_span_tree_2(v_disp_matrix, h_disp_matrix, q_v_disp_matrix, q_h_disp_matrix)
% Original code modified from: https://github.com/usnistgov/MIST/tree/mist-matlab
% Ref: https://isg.nist.gov/deepzoomweb/resources/csmet/pages/image_stitching/image_stitching.html

% Constants for direction instructions
MST_START_TILE = 10;

[nb_vertical_tiles, nb_horizontal_tiles] = size(v_disp_matrix);
% Initialize the starting point of each image. These starting points are computed relative to Matlab image coordinate system ==>
% x is left to right and y is up to down. global_y_img_pos is vertical or y and global_x_img_pos is horizontal or x
global_z_img_pos = zeros(nb_vertical_tiles,nb_horizontal_tiles);

% Initialize the tiling indicator matrix that gives us the direction by which images were stitched 
% in the vertical direction up 11, down 12
% in the horizontal direction right 21, left 22
% This means that if an element of tiling indicator (i,j) has 11 in it, that means that this tile was stitched to the one above it (i-1,j) and
% if an element has 21 in it that means that this tile was stitched to the one on its right (i,j+1) in the global image 
tiling_indicator = zeros(nb_vertical_tiles,nb_horizontal_tiles);

[val1, indx1] = max(q_v_disp_matrix(:));
[val2, indx2] = max(q_h_disp_matrix(:));
if val1 > val2
    [ii, jj] = ind2sub([nb_vertical_tiles, nb_horizontal_tiles], indx1);
else
    [ii, jj] = ind2sub([nb_vertical_tiles, nb_horizontal_tiles], indx2);
end
tiling_indicator(ii,jj) = MST_START_TILE;

% Compute tiles positions
% correlations are inverted because for this application we actually want a maximum spanning tree
[tiling_indicator, global_z_img_pos, tiling_coeff] = minimum_spanning_tree_worker(tiling_indicator, global_z_img_pos,...
    v_disp_matrix, h_disp_matrix, -q_v_disp_matrix, -q_h_disp_matrix); 

tiling_coeff = -tiling_coeff;
tiling_coeff(ii,jj) = 5; % the starting point

tiling_coeff(~tiling_indicator) = NaN;
global_z_img_pos(~tiling_indicator) = NaN;
tiling_indicator(~tiling_indicator) = NaN;

% translate the positions to (1,1)
global_z_img_pos = global_z_img_pos - max(global_z_img_pos(:)) + 1;


end


function [tiling_indicator, global_z_img_pos, tiling_coeff] = minimum_spanning_tree_worker(tiling_indicator, global_z_img_pos, v_disp_matrix, h_disp_matrix, q_v_disp_matrix, q_h_disp_matrix)

% Constants for direction instructions
MST_CONNECTED_NORTH = 11;
MST_CONNECTED_SOUTH = 12;
MST_CONNECTED_LEFT = 22;
MST_CONNECTED_RIGHT = 21;

% Initialize the minimum spanning tree value and set the first value in tiling_coeff to the highest = -1.
[tile_y, tile_x] = size(v_disp_matrix);
mst_value = 0;
tiling_coeff = zeros(tile_y, tile_x);
tiling_coeff(tiling_indicator > 0) = -1;

% Keep on finding the next vertice in the tree until all is found. The first vertice is always the position of the first image 
% defined in global_y_img_pos(1,1) and global_x_img_pos(1,1)
for j = 2:numel(tiling_indicator)
    
    % Check the vertices that are already connected to the tree
    [I, J] = find(tiling_indicator);
    
    % Initialize the minimum coefficient value
    mst_min = Inf;
    
    % Scan all the unconnected neighbors of the connected vertices and add the one with the lowest correlation coefficient to the tree
    for i = 1:length(I)
        % Check the neighbor below
        % Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
        if I(i)<tile_y && tiling_indicator(I(i)+1,J(i)) == 0 && q_v_disp_matrix(I(i)+1,J(i)) < mst_min
            % update the minimum coefficient value
            mst_min = q_v_disp_matrix(I(i)+1,J(i));
            stitching_index = MST_CONNECTED_NORTH; % index that indicates the stitching direction of the minimal coefficient
            mst_i = I(i);
            mst_j = J(i);
        end
        
        % Check the neighbor above
        % Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
        if I(i)>1 && tiling_indicator(I(i)-1,J(i)) == 0 && q_v_disp_matrix(I(i),J(i)) < mst_min
            % update the minimum coefficient value
            mst_min = q_v_disp_matrix(I(i),J(i));
            stitching_index = MST_CONNECTED_SOUTH; % index that indicates the stitching direction of the minimal coefficient
            mst_i = I(i);
            mst_j = J(i);
        end
        
        % Check the neighbor to the right
        % Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
        if J(i)<tile_x && tiling_indicator(I(i),J(i)+1) == 0 && q_h_disp_matrix(I(i),J(i)+1) < mst_min
            % update the minimum coefficient value
            mst_min = q_h_disp_matrix(I(i),J(i)+1);
            stitching_index = MST_CONNECTED_LEFT; % index that indicates the stitching direction of the minimal coefficient
            mst_i = I(i);
            mst_j = J(i);
        end
        
        % Check the neighbor to the left
        % Check if it is valid and isn't already stitched and that the correlation coefficient is minimal
        if J(i)>1 && tiling_indicator(I(i),J(i)-1) == 0 && q_h_disp_matrix(I(i),J(i)) < mst_min
            % update the minimum coefficient value
            mst_min = q_h_disp_matrix(I(i),J(i));
            stitching_index = MST_CONNECTED_RIGHT; % index that indicates the stitching direction of the minimal coefficient
            mst_i = I(i);
            mst_j = J(i);
        end
    end
    
    % update the minimum spanning tree value and the tiling coefficient
    mst_value = mst_value + mst_min;
    
    % Compute the starting position of the chosen tile
    % Check the neighbor below
    if stitching_index == MST_CONNECTED_NORTH
        global_z_img_pos(mst_i+1,mst_j) = global_z_img_pos(mst_i,mst_j) + v_disp_matrix(mst_i+1,mst_j);
        
        % update tiling indicator
        tiling_indicator(mst_i+1,mst_j) = MST_CONNECTED_NORTH;
        tiling_coeff(mst_i+1,mst_j) = mst_min;
    end
    
    % Check the neighbor above
    if stitching_index == MST_CONNECTED_SOUTH
        global_z_img_pos(mst_i-1,mst_j) = global_z_img_pos(mst_i,mst_j) - v_disp_matrix(mst_i,mst_j);
        
        % update tiling indicator
        tiling_indicator(mst_i-1,mst_j) = MST_CONNECTED_SOUTH;
        tiling_coeff(mst_i-1,mst_j) = mst_min;
    end
    
    % Check the neighbor to the right
    if stitching_index == MST_CONNECTED_LEFT
        global_z_img_pos(mst_i,mst_j+1) = global_z_img_pos(mst_i,mst_j) + h_disp_matrix(mst_i,mst_j+1);
       
        % update tiling indicator
        tiling_indicator(mst_i,mst_j+1) = MST_CONNECTED_LEFT;
        tiling_coeff(mst_i,mst_j+1) = mst_min;
    end
    
    % Check the neighbor to the left
    if stitching_index == MST_CONNECTED_RIGHT
        global_z_img_pos(mst_i,mst_j-1) = global_z_img_pos(mst_i,mst_j) - h_disp_matrix(mst_i,mst_j);
        
        % update tiling indicator
        tiling_indicator(mst_i,mst_j-1) = MST_CONNECTED_RIGHT;
        tiling_coeff(mst_i,mst_j-1) = mst_min;
    end
end

end




