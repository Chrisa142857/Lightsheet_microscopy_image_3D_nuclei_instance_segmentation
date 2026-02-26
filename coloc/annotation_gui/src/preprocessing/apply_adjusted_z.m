function path_table = apply_adjusted_z(path_table, z_disp_matrix)
%--------------------------------------------------------------------------
% Calculated adjusted z positions from adjustment matrix.
%--------------------------------------------------------------------------

% Start from lowest z position 
%c = min(path_table.z)-1;
%path_table.z_adj = path_table.z-c;
path_table.z_adj = path_table.z;

% Apply z adjustments from adjustment matrix
[nrows, ncols] = size(z_disp_matrix);
for i = 1:nrows
    for j = 1:ncols
        idx = path_table.y == i & path_table.x == j;
        path_table(idx,:).z_adj = path_table(idx,:).z_adj - z_disp_matrix(i,j);
    end
end

% Remove tiles where not all z positions are present
v = sum(path_table.z_adj==path_table.z_adj');
path_table(v ~= max(v),:) = [];

% Set the lowest adjusted z value to z = 1
%if min(path_table.z_adj) < 1
    path_table.z_adj = path_table.z_adj-min(path_table.z_adj)+1;
%end
z_adj = [path_table(:,1) path_table(:,end)];

% Attach to path_table
[~,z_idx] = setdiff(path_table.file,z_adj.file);
path_table(z_idx,:) = []; 
path_table.z_adj = z_adj.z_adj;

end