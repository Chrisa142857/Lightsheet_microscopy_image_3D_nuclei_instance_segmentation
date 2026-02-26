function measure_full_thickness(input)

% Load path to results summary
if isstring(input) || ischar(input)
    config = NM_config('evaluate',input);
    res_path = config.results_path;
elseif ~isfield(input,'stage')
    error("Requires configuration structure or sample id")
elseif isequal(input.stage,'analyze')
    res_path = fullfile(input.output_directory,strcat(input.sample_id,'_results.mat'));
else
    res_path = input.results_path;
end

% Load and preprocess atlas boundary points
home_path = fileparts(which('NM_config'));
load(fullfile(home_path,'data','annotation_data','flatviewCortex.mat'),'thickness')
img_size = [456,320,528];
l1_idx = thickness.l1_left_idx;
l6_idx = thickness.l6_left_idx;
l1_idx = l1_idx(l1_idx>0);
l6_idx = l6_idx(l6_idx>0);
[y1,x1,z1] = ind2sub(img_size,l1_idx);
[y6,x6,z6] = ind2sub(img_size,l6_idx);
img_size = [228,320,528];
l1_sub = permute_points([y1,x1,z1],img_size,'lsa','ail',false,false);
l6_sub = permute_points([y6,x6,z6],img_size,'lsa','ail',false,true);

load(fullfile(home_path,'data','annotation_data','flatviewCortex.mat'),'voxelmap_10')
voxelmap_10 = voxelmap_10(1:1360,:);

% Measure thickness of samples
[~,samples] = arrayfun(@(s) fileparts(s),res_path);
n_samples = length(samples);

for i = 1:n_samples    
    if isequal(config.groups{i}(:,2),"FW")
        continue
    end
    fprintf("Working on %s\n",samples(i))

    
    % Load registration parameters mapping atlas to image
    if isfile(res_path(i))
        load(res_path(i),'reg_params')
        atlas_size = reg_params.atlas_img_size;
    else
        error("Could not locate result summary %s",res_path(i))
    end
    
    if ~isfield(reg_params,'atlas_to_image')
        warning("Atlas to image registration has not been mapped for this sample")
        continue
    else
        reg_params = reg_params.image_to_atlas;
    end
    
    % Transform atlas boundary points to match image    
    s1_idx = transformix(l1_sub(:,[2,1,3]),reg_params,[1,1,1],[]);
    s1_idx = s1_idx(:,[2,1,3]);
    s1_idx = round(s1_idx);
    
    s6_idx = transformix(l6_sub(:,[2,1,3]),reg_params,[1,1,1],[]);
    s6_idx = s6_idx(:,[2,1,3]);
    s6_idx = round(s6_idx);

    % Measure distance to between layer 1 voxel and the nearest layer 6
    % voxel
    fprintf("Calculating boundary distances...\n")
    [s1u,~,s1u_idx] = unique(s1_idx,'rows');
    distance = zeros(1,size(s1u,1));
    for j = 1:length(distance)
       p = s1u(j,:);
       distance(j) = min(sqrt((s6_idx(:,1)-p(1)).^2 + (s6_idx(:,2)-p(2)).^2 + (s6_idx(:,3)-p(3)).^2));
    end
    distance_full = distance(s1u_idx);
    
    % Save to results structure
    thickness = distance_full;
    save(res_path(i),'-append','thickness')
    
    % Testing
    %[~,~,c] = unique(voxelmap_10);
    %d2 = distance_full(c);
    %d2 = reshape(d2,size(voxelmap_10));
    %d2(voxelmap_10==1) = 0;
    
    %img_size = [528,320,228];
    %img_size = [373,296,213];
    %s1_idx = s1_idx(s1_idx(:,1)<img_size(1),:);
    %s1_idx = s1_idx(s1_idx(:,2)<img_size(2),:);
    %s1_idx = s1_idx(s1_idx(:,3)<img_size(3),:);
    %img0 = zeros(img_size,'uint8');
    %v1 = sub2ind(img_size,s1_idx(:,1),s1_idx(:,2),s1_idx(:,3));
    %v1 = sub2ind(img_size,l1_sub(:,1),l1_sub(:,2),l1_sub(:,3));
    %img0(v1) = 255;
    %niftiwrite(img0,'test.nii')
end


end