function fm = convert_to_flatmap(img,combine_method)

if nargin<2
    combine_method = 'mean';
elseif ~ismember(combine_method,{'mean','sum','max'})
    error("Invalid voxel merging function")
end

home_path = fileparts(which('NM_config'));
fc_path = load(fullfile(home_path,'data','annotation_data','flatviewCortex.mat'));
voxelmap = fc_path.voxelmap_100;
voxelpaths = fc_path.voxelpaths_100;

if length(size(img)) == 4
    img = squeeze(img);
end

img = cat(3,img,zeros(size(img)));
img = permute(img,[3,2,1]);
img = flip(img,2);
img = flip(img,1);
results = zeros(1,length(voxelpaths));
for j = 1:length(voxelpaths)
    c = voxelpaths(:,j);
    
    if isequal(combine_method,'mean')
        results(j) = mean(img(c(c~=0)));
    elseif isequal(combine_method,'sum')
        results(j) = sum(img(c(c~=0)));
    else
        results(j) = max(img(c(c~=0)));
    end
end

fm = zeros(1,length(voxelmap(:)));
for j = 1:length(fm)
    if voxelmap(j)>1
        fm(j) = results(voxelmap(j));
    end
end

fm = reshape(fm,2720,1360);
fm = fm(1361:end,:);
fm = imrotate(fm,-90);
end