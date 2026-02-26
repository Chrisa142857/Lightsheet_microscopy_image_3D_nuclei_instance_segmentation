function calc_flatmap_stats(config)
%--------------------------------------------------------------------------
% Create 4D voxel volume representing fold change, p, q values
%--------------------------------------------------------------------------

fdr_thresh = 0.05;
smoothing_radius = 12;

% Load structure/volume data
n_samples = length(config.results_path);
measure_vars = {'thickness','surface_area','volume'};
imgs = cell(n_samples,length(measure_vars));
for i = 1:length(imgs)
    var_names = who('-file',config.results_path(i));
    vars_sub = measure_vars(ismember(measure_vars,var_names));
    for j = 1:length(vars_sub)
        data = load(config.results_path(i),vars_sub{j});
        data = data.(vars_sub{j});
        imgs{i,j} = data;
    end    
end
empty_idx = all(cellfun(@(s) isempty(s),imgs));

% Assign samples to groups
group_delimiters = cat(2,config.samples,cat(1,config.groups{:}));
set1 = group_delimiters(group_delimiters(:,3) == config.compare_groups(1),1);
vox1 = imgs(ismember(config.samples,set1));
set2 = group_delimiters(group_delimiters(:,3) == config.compare_groups(2),1);
vox2 = imgs(ismember(config.samples,set2));

% Load voxel map
fc_path = load(fullfile(config.home_path,'data','annotation_data','flatviewCortex.mat'));
voxelmap = fc_path.voxelmap_10(1:1360,:);
[~,~,c] = unique(voxelmap);

% Do cell counts after reading directly from images
vox_files = dir(fullfile(config.vox_directory,"*.nii"));
vox_files = fullfile(config.vox_directory,{vox_files.name});
h = fspecial('disk',smoothing_radius);
vox_files = [];
for i = 1:length(vox_files)
    img = read_img(vox_files(i));
    img1 = cell(1,3);
    for j = 1:size(img,4)
        s = convert_to_flatmap(squeeze(img(:,:,:,j)));
        if j == 3
            s(s<1.3010) = 1.3010;
        end
        img1{j} = imfilter(s,h,'replicate');
    end
    img1 = cat(3,img1{:});
    
    % Save file
    [~,b] = fileparts(vox_files(i));
    b = strsplit(b,'_');
    fname = fullfile(config.flat_directory,sprintf("%s_%s_flatmap.nii",...
        config.prefix,b(2)));
    niftiwrite(img1,fname)
end


% Do volume, surface area, thickness
for i = 1:length(measure_vars)
    if empty_idx(i)
        continue
    else
        vox_res = repmat({zeros(size(voxelmap))},1,3);
    end

    % Generate flatmaps
    for j = 1:length(vox1)
        d2 = vox1{j}(c);
        d2 = reshape(d2,size(voxelmap));
        d2(voxelmap==1) = 0;
        vox1{j} = d2;
    end
    for j = 1:length(vox2)
        d2 = vox2{j}(c);
        d2 = reshape(d2,size(voxelmap));
        d2(voxelmap==1) = 0;
        vox2{j} = d2;
    end
    
    % Normalize to to average of each sample
    vox1 = cellfun(@(s) s/mean(s(s>0)), vox1, 'UniformOutput', false);
    vox2 = cellfun(@(s) s/mean(s(s>0)), vox2, 'UniformOutput', false);
    
    % Calculate means
    m_set1 = mean(cat(3,vox1{:}),3);
    m_set2 = mean(cat(3,vox2{:}),3);
    idx = max(cat(3,m_set1,m_set2),[],3)>0;
        
    % Relative change
    vox_res{1}(idx) = (m_set2(idx)-m_set1(idx))./m_set1(idx);    
    
    % Bin to 100um before calculating p-values    
    vox1 = cellfun(@(s) imresize(s,0.1,'Method','bilinear'),vox1,'UniformOutput',false);
    vox2 = cellfun(@(s) imresize(s,0.1,'Method','bilinear'),vox2,'UniformOutput',false);

    down_size = size(vox1{1});
    vox1 = cellfun(@(s) reshape(s,numel(s),1),vox1,'UniformOutput',false);
    vox2 = cellfun(@(s) reshape(s,numel(s),1),vox2,'UniformOutput',false);
    
    % p value
    [~,p] = ttest2(cat(2,vox1{:})',cat(2,vox2{:})');    
    na_idx = isnan(p);
    adj_p = zeros(1,length(p));
    [~,~,~, q] = fdr_bh(p(~na_idx));
    adj_p(~na_idx) = q;
    p(na_idx) = 0.05;
    adj_p(na_idx) = 0.05;
    
    % Resize and save 
    p = reshape(p,down_size);
    adj_p = reshape(adj_p,down_size);
    p = imresize(p,size(voxelmap),'Method','bilinear');
    adj_p = imresize(adj_p,size(voxelmap),'Method','bilinear');

    %h = fspecial('disk',smoothing_radius);
    %p = imfilter(p,h,'replicate');
    %adj_p = imfilter(adj_p,h,'replicate');
    p(~idx) = 1;
    adj_p(adj_p>fdr_thresh) = 1;    
    vox_res{2} = -log10(p);
    vox_res{3} = -log10(adj_p);
    
    % Adjust orientation
    vox_res = cellfun(@(s) flip(s,1),vox_res,'UniformOutput',false);
    vox_res = cellfun(@(s) imrotate(s,-90),vox_res,'UniformOutput',false);
    vox_res = cat(3,vox_res{:});
    
    % Save file
    fname = fullfile(config.flat_directory,sprintf("%s_%s_flatmap.nii",...
        config.prefix,measure_vars{i}));
    niftiwrite(vox_res,fname)
end


end


