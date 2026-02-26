function get_structure_patch(config,index,patch_size,offset)
% Get centroid patch at annotation index

register_resolution = 25;

if nargin<3
    patch_size = [50,100,100]; 
end

if nargin<4
    offset = [0,0,0];
end

plot_classes = 'all';

% Check for centroids
res_path = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));
if ~isfile(res_path)
    error("Could not locate results structure in %s output directory",config.sample_id)
else
    var_names = who('-file',res_path);
    if ismember('I_mask',var_names)
        load(res_path,'I_mask')
    else
        error("No annotation mask detect in results structure")
    end
    if ismember('centroids',var_names)
        load(res_path,'centroids')
    else
        error("No centroids detected in results structure")
    end
    if ~isequal(plot_classes,'none')
        if ismember('classes',res_path)
            load(res_path,'classes')
            assert(length(classes) == size(centroids,1),...
                "Number of centroids and classes do not match")
        else
            warning("No classes detected in results structure")
        end
    end
end

% Bin annotations for specified index
I_mask = bin_annotation_structures(I_mask,index);

% Get centroid position of binary mask
cen = regionprops3(I_mask>0,'Centroid','Volume');
a = sum(cen.Centroid.*cen.Volume,1);
cen = a./sum(cen.Volume);
disp(round(cen))

cen2 = regionprops(I_mask(:,:,round(cen(3)))>0,'Centroid','Area');
disp(cen2)
%a = sum(cen2.Centroid.*cen2.Area,1);
%cen2 = a./sum(cen2.Volume);
cen2 = cen2.Centroid;
cen = [cen2,cen(3)];
cen = [66,200,42];

% Adjust centroid coordinate position
res_adj = repmat(register_resolution,1,3)./config.resolution{1};
cen = cen([3,2,1]).*res_adj([3,1,2]) + offset;

% Save centroid patch to samples
plot_centroid_patch(config, cen, patch_size);

end

