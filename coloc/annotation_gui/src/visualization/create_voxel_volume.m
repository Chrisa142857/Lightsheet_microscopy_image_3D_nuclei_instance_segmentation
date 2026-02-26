function results = create_voxel_volume(results, bin_resolution)
% Create single binned voxel volume from results structure

registration_resolution = 25;

if isfield(results.sample_data,'resolution')
    res = results.sample_data.resolution;
else
    res = [1.21,1.21,4];
end

if nargin<2
    bin_resolution = 100;
end

% Re-map coordinates and transform
res_adj = res/registration_resolution;
centroids = round(results.centroids.*res_adj,2);
centroids = centroids(:,[2,1,3]);
fprintf("Transforming centroid coordinates...\n")
centroids = transformix(centroids,results.reg_params.atlas_to_image,[1,1,1],[]);
centroids = centroids(:,[2,1,3]);

res_adj = 25/bin_resolution;
centroids = round(centroids*res_adj);

% Size of atlas at 25um/voxel
img_size = [528,320,228];
img_size = round(img_size*res_adj);

% Remove centroids falling out of range and linearize
rm_idx = any(centroids<1,2);
rm_idx = rm_idx | centroids(:,1)>img_size(1) | centroids(:,2)>img_size(2) | centroids(:,3)>img_size(3);
cen = centroids(~rm_idx,:);
cen = sub2ind(img_size,cen(:,1),cen(:,2),cen(:,3));

% If classes present, make image for class
if isfield(results,'classes')
    classes = results.classes(~rm_idx);
    unique_classes = unique(results.classes);
else
    classes = ones(size(cen,1),1);
    unique_classes = 1;
end
   
% Create voxel image for each class
img = cell(1,length(unique_classes));
for i = 1:length(unique_classes)
    cen_sub = cen(classes == unique_classes(i),:);

    % Count number of instances in each voxel
    [cnt_unique, unique_a] = hist(cen_sub,unique(cen));

    % Assign counts as voxel intensities and reshape
    v = zeros(1,img_size(1)*img_size(2)*img_size(3));
    v(unique_a) = cnt_unique;
    img{i} = reshape(v,img_size(1),img_size(2),img_size(3));
end

% Attach to results structure
results.voxel_volume = img;

end

