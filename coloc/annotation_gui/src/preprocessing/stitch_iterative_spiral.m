function stitch_iterative_spiral(config, path_table)

% Default position order: left, right, top, bottom

% Get tile information
nrows = length(unique(path_table.y));
ncols = length(unique(path_table.x));

% Check for adjusted z positions
if ~isfield(path_table,'z_adj')
    z_disp_matrix = load_var(config, 'z_disp_matrix');
    assert(nrows == size(z_disp_matrix,1) && ncols == size(z_disp_matrix,2), "Loaded z adjustment parameters do not match number of tiles "+...
        "detected in the input image directory. Check input image directory or update z adjustment")
    path_table = apply_adjusted_z(path_table, z_disp_matrix);
end

% Calculate tile position order
[x, y] = spiral(nrows,ncols);
positions = [x - min(x) + 1; y - min(y) + 1];

% Calculate overlapping regions for each tile
tempI = read_img(path_table);
[img_height, img_width] = size(tempI);
z_adj = path_table.z_adj;

h_overlap = ceil(img_width*config.overlap);
v_overlap = ceil(img_height*config.overlap);
overlaps{1} = {1:img_height, 1:h_overlap}; % left
overlaps{2} = {1:img_height, img_width-h_overlap+1:img_width}; % right
overlaps{3} = {1:v_overlap, 1:img_width}; % top
overlaps{4} = {img_height-v_overlap+1:img_height, 1:img_width}; % bottom

% Generate image name grid
img_name_grid = cell(max(path_table.y),max(path_table.x),...
    length(unique(path_table.markers)),length(unique(z_adj)));
path_table = sortrows(path_table,["z_adj","channel_num","x","y"],'ascend');

% Arrange images into correct positions
try
    img_name_grid = reshape(path_table.file,size(img_name_grid));
catch ME
    if isequal(ME.identifier,'MATLAB:getReshapeDims:notSameNumel')
        error("Inconsistent image file information. Recalculate adjusted z and/or check configuration")
    end
end

% Calculate transforms
stitch_tforms = zeros(size(positions,2)-1,size(img_name_grid,4));
for i = 1:length(z_adj)
    img_grid = squeeze(img_name_grid(:,:,1,i));
    stitch_tforms(i,:) = stitch_spiral_worker(img_grid, positions, overlaps);
end

% Apply transforms and merge images



end

function tforms = stitch_spiral_worker(img_grid, positions, overlaps)

a = 1;


for i = 1:length(x)-1
    if x(i+1) > x(i)
        disp('right')
    elseif x(i+1) < x(i)
        disp('left')
    end
    if y(i+1) > y(i)
        disp('up')
    elseif y(i+1) < y(i)
        disp('down')
    end
end


tforms = 1;

end


function [x, y] = spiral(X, Y)
% Get tile positions in the shape of a spiral
% Method from: https://stackoverflow.com/questions/398299/looping-in-a-spiral

xx = 0;
yy = 0;
d = [0, -1];

idx = 1;
for i = 1:max(X,Y)^2
    if (-X/2 < xx && xx <= X/2) && (-Y/2 < yy && yy <= Y/2)
        x(idx) = xx;
        y(idx) = yy;
        idx = idx+1;
    end
    if xx == yy || (xx < 0 && xx == -yy) || (xx > 0 && xx == 1-yy)
        d = [-d(2), d(1)];
    end
    xx = xx+d(1);
    yy = yy+d(2);
end

end