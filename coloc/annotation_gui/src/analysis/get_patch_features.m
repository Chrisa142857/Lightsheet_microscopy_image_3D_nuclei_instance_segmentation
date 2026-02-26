function stable = get_patch_features(centroids,path_table,config)

% Get markers to measure
s = config.patch_size(2);
markers = config.markers(config.classify_channels);

% Get patch features for remaining centroids
z_pos = unique(centroids.coordinates(:,3));
stable = cell(length(z_pos),length(markers));
for i = 1:length(z_pos)
    fprintf("Working on z position %d \n",z_pos(i))
    idx = centroids.coordinates(:,3) == z_pos(i);
    cen_sub = centroids.coordinates(idx,:);
    
    % Load images and get features
    for j = 1:length(markers)
        file = path_table(path_table.z == z_pos(i) &...
            path_table.markers == markers(j),:);
        file = file.file{1};
        f = zeros([size(cen_sub,1),8]);
        yx = cen_sub(:,[1,2]);
        parfor k = 1:size(yx,1)
            try 
                pos = yx(k,:);
                ranges = {[pos(1)-s,pos(1)+s], [pos(2)-s,pos(2)+s]};
                img = imread(file,'PixelRegion',ranges);
                f(k,:) = measure_patch_features(img, s, false);
            catch
            end
        end
        stable{i,j} = f;
    end
end
stable = cell2mat(stable);
end