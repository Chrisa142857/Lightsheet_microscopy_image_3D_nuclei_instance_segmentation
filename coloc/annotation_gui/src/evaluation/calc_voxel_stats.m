function calc_voxel_stats(config)
%--------------------------------------------------------------------------
% Create 4D voxel volume representing fold change, p, q values
%--------------------------------------------------------------------------

fdr_thresh = 0.05;
generate_flatmaps = true;

% Load voxel volumes
n_samples = length(config.results_path);
imgs = cell(1,n_samples);
f_imgs = cell(1,n_samples);
for i = 1:length(imgs)
    var_names = who('-file',config.results_path(i));
    if ~ismember('voxel_volume',var_names)
        error("No voxel volume found for sample %s",config.samples(i))
    end
    load(config.results_path(i),'voxel_volume')
    imgs{i} = voxel_volume(config.keep_classes);
    imgs{i} = cellfun(@(s) s(:),imgs{i},'UniformOutput',false);
    if generate_flatmaps
        f_imgs{i} = cellfun(@(s) convert_to_flatmap(s,'sum'),voxel_volume(config.keep_classes),'UniformOutput',false);
        f_imgs{i} = cellfun(@(s) s(:),f_imgs{i},'UniformOutput',false);
    end
end

% Sum all classes
sums = {};
f_sums = {};
if isequal(config.sum_all_classes,"true")
    config.class_names = ["all",config.class_names];
    sums = cell(1,length(imgs));
    for i = 1:length(sums)
        sums{i} = sum(cat(4,imgs{i}{:}),4);
    end
    if generate_flatmaps
        f_sums = cell(1,length(f_imgs));
        for i = 1:length(sums)
            f_sums{i} = sum(cat(4,f_imgs{i}{:}),4);
        end
    end
end

% Add custom classes
cust = {};
f_cust = {};
if ~isempty(config.custom_class)
    n_single = length(config.class_names(config.keep_classes));
    cust = cell(1,length(imgs));    
    for i = 1:length(imgs)
        counts = cellfun(@(s) s(:),imgs{i},'UniformOutput',false);
        cust{i} = get_custom_class(cat(2,counts{:}),1:n_single,config.custom_class);
        cust{i}(isnan(cust{i})) = 0;
    end
    if generate_flatmaps
        f_cust = cell(1,length(f_imgs));
        for i = 1:length(f_imgs)  
            counts = cellfun(@(s) s(:),f_imgs{i},'UniformOutput',false);
            f_cust{i} = get_custom_class(cat(2,counts{:}),1:n_single,config.custom_class);
            f_cust{i}(isnan(f_cust{i})) = 0;
        end
    end
end

imgs = cellfun(@(r,s,t) cell2mat(cat(2,r,s,t)), sums, imgs, cust, 'UniformOutput', false);
f_imgs = cellfun(@(r,s,t) cell2mat(cat(2,r,s,t)), f_sums, f_imgs, f_cust, 'UniformOutput', false);

% Assign samples to groups
group_delimiters = cat(2,config.samples,cat(1,config.groups{:}));
set1 = group_delimiters(group_delimiters(:,3) == config.compare_groups(1),1);
vox1 = imgs(ismember(config.samples,set1));
f_vox1 = f_imgs(ismember(config.samples,set1));

set2 = group_delimiters(group_delimiters(:,3) == config.compare_groups(2),1);
vox2 = imgs(ismember(config.samples,set2));
f_vox2 = f_imgs(ismember(config.samples,set2));

img_sizes = size(voxel_volume{1});

% For each class, generate vox
n_classes = size(imgs{1},2);
config.class_names = arrayfun(@(s) strrep(s,'./',''),config.class_names);

% Get samples
for i = 1:n_classes
    vox_res = ones(prod(img_sizes),3);
    fvox_res = ones(1360*1360,3);
    
    % Seperate groups
    set1 = cat(2,cell2mat(cellfun(@(s) s(:,i),vox1,'UniformOutput',false)));
    set2 = cat(2,cell2mat(cellfun(@(s) s(:,i),vox2,'UniformOutput',false)));
    
    f_set1 = cat(2,cell2mat(cellfun(@(s) s(:,i),f_vox1,'UniformOutput',false)));
    f_set2 = cat(2,cell2mat(cellfun(@(s) s(:,i),f_vox2,'UniformOutput',false)));
    
    % Calculate means
    m_set1 = mean(set1,2);
    m_set2 = mean(set2,2);
    m_fset1 = mean(f_set1,2);
    m_fset2 = mean(f_set2,2);
    
    % Threshold counts
    if max([f_set1,f_set2],[],'all') <=1
        idx = find(m_set1>0.03 & m_set2>0.03);
       
       % Threshold counts
        fidx = find(m_fset1>0.03 & m_fset2>0.03);
    else
        idx = find(m_set1>config.minimum_cell_number &...
           m_set2>config.minimum_cell_number);
       
           % Threshold counts
        fidx = find(m_fset1>config.minimum_cell_number &...
           m_fset2>config.minimum_cell_number);
    end

   % Fold change
    vox_res(:,1) = 0;
    vox_res(idx,1) = (m_set2(idx)-m_set1(idx))./m_set1(idx); 

   % Fold change
    fvox_res(:,1) = 0;
    fvox_res(fidx,1) = (m_fset2(fidx)-m_fset1(fidx))./m_fset1(fidx); 

    % p value
    [~,p] = ttest2(set1(idx,:)',set2(idx,:)');
    vox_res(idx,2) = -log10(p);
    
    [~,fp] = ttest2(f_set1(fidx,:)',f_set2(fidx,:)');
    fvox_res(fidx,2) = -log10(fp);

    % Adjusted p value. Apply threshold to adjusted p value
    [~,~,~, adj_p] = fdr_bh(p);
    adj_p = -log10(adj_p);
    adj_p(adj_p<-log10(fdr_thresh)) = -log10(fdr_thresh);
    vox_res(idx,3) = adj_p;
    
    [~,~,~, adj_fp] = fdr_bh(fp);
    adj_fp = -log10(adj_fp);
    adj_fp(adj_fp<-log10(fdr_thresh)) = -log10(fdr_thresh);
    fvox_res(fidx,3) = adj_fp;
    
    % Reshape and save to file
    vox_res = reshape(vox_res,[img_sizes,3]);
    fvox_res = reshape(fvox_res,[1360,1360,3]);
    
    % Save file
    fname = fullfile(config.vox_directory,sprintf("%s_%s_voxels.nii",...
        config.prefix,config.class_names(i)));
    niftiwrite(vox_res,fname)
    
    fname = fullfile(config.flat_directory,sprintf("%s_%s_flatmap.nii",...
        config.prefix,config.class_names{i}));
    niftiwrite(fvox_res,fname)
end

end