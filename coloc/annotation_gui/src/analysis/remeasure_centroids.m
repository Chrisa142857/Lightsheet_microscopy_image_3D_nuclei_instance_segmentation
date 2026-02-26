function [m_int, m_back] = remeasure_centroids(centroids,path_table,config,markers)
%--------------------------------------------------------------------------
% Remeasure centroid intensities from centroid list.
%--------------------------------------------------------------------------
res = config.resolution{1}(1); % Voxel resolution
method = 'mean';        % How to measure centroids. mean or max
s = 1;  % Sampling neighbors. If set to 1, this will sample the centroid + 8 neighbors
sub_back = 'false';  % Option to subtract background

tempI = loadtiff(path_table.file{1});
[nrows, ncols] = size(tempI);

z_pos = unique(centroids(:,3));
m_int = zeros(length(markers),size(centroids,1));
m_back = zeros(length(markers),size(centroids,1));
parfor i = 1:length(z_pos)
    fprintf('%s\t Remeasuring image %d \n',datetime('now'),z_pos(i));
    
    new_vals = zeros(length(markers),size(centroids,1));
    new_back_vals = zeros(length(markers),size(centroids,1));
    
    idx = centroids(:,3)==z_pos(i);
    
    % Get linear indexes for cooridnates
    y = centroids(idx,1);
    x = centroids(idx,2);

    y1 = arrayfun(@(a,b) meshgrid(a-s:a+s,b-s:b+s),y,x,'UniformOutput',false);
    x1 = arrayfun(@(a,b) meshgrid(a-s:a+s,b-s:b+s)',x,y,'UniformOutput',false);

    % Remove coordinates outside of bounds of images
    for j = 1:length(x1)
       rm_idx = y1{j} < 1 | y1{j} > nrows;
       rm_idx = rm_idx |  x1{j} < 1 | x1{j} > ncols;
       y1{j} = y1{j}(~rm_idx);
       x1{j} = x1{j}(~rm_idx);
    end

    % Linearize
    idxs = cellfun(@(a,b) sub2ind([nrows,ncols],a,b),y1,x1,...
        'UniformOutput',false);

    % Read image and measure intensity
    for j = 1:length(markers)
        zmask = path_table.z == z_pos(i) &...
            path_table.markers == markers(j);
        if ~any(zmask)
            continue
        end
        img_file = path_table(zmask,:);  
    
        I = loadtiff(img_file.file{1});
    
        % Get values at indexes
        if isequal(method,'mean')
            vals = cellfun(@(k) mean(I(k)),idxs);
        else
            vals = cellfun(@(k) max(I(k)),idxs);
        end
        
        % Save values
        new_vals(j,idx) = vals';
        
        % Optional: measure background
        if isequal(sub_back,'true')
            [~, B] = smooth_background_subtraction(I,'false',res*50);
            B = imgaussfilt(B,20);
            vals_back = cellfun(@(k) mean(B(k)),idxs);
            % Save values
            new_back_vals(j,idx) = vals_back';
        end
    end
    
    % Add to final matrix
    m_int = m_int + new_vals;
    if isequal(sub_back,'true')
        m_back = m_back + new_back_vals;
    end
end

m_int = m_int';

if isequal(sub_back,'true')
    save_path = fullfile(config.output_directory, 'background');
    if exist(save_path,'dir') ~= 7
        mkdir(save_path);
    end
    writematrix(m_back',fullfile(save_path,'back_values.csv'))
end

end
