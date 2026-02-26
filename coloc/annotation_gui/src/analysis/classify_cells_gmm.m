function [ct, p, gm] = classify_cells_gmm(centroids, config)
%--------------------------------------------------------------------------
% Classify cell-types using a Gaussian Mixture Model of fluorescence
% intensity
%--------------------------------------------------------------------------
remove_l1 = "true";             % Remove cortical layer 1 cells

% Get parameters from config structure
n_clusters = config.n_clusters;
confidence = config.confidence;
markers = config.markers;

n_markers = length(markers);

if n_clusters < n_markers
    n_clusters = repmat(n_clusters,1,n_markers);
end

% Create empty vector for storing cell-types
ct = ones(size(centroids,1),1);

% Remove any centroids without annotations
rm_idx = centroids(:,4) == 0;
fprintf('%s\t Removed %d nuclei with no annotation \n',datetime('now'),sum(rm_idx))

% Remove cells below minimum nuclei intensity threshold
low_idx = centroids(:,5) < config.min_intensity;

rm_idx = rm_idx | low_idx;
fprintf('%s\t Removed %d low intensity nuclei \n',datetime('now'),sum(rm_idx))

% Ask if to remove layer 1 prior to gmm analysis
if isequal(remove_l1,"true")
   l1_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer1.csv'));
   l1_idx = ismember(centroids(:,4),l1_table.index);
   
   s1 = centroids(:,5) > prctile(centroids(:,5),90);
   s2 = centroids(:,6) > prctile(centroids(:,6),90);
   s3 = centroids(:,7) > prctile(centroids(:,7),90);

   l1_idx = s1 & s2 & s3;% | l1_idx;
   
   rm_idx = rm_idx | l1_idx;
   fprintf('%s\t Removed %d cells \n',datetime('now'),sum(l1_idx))
else
    l1_idx = 0;
end

% Remove selected indexes
centroids = centroids(~rm_idx,:);
annotations = centroids(:,4);
layers = get_structure_layers(annotations,config);

% Intitialize matrices and models
gm = cell(1,n_markers-1);
p = cell(1,n_markers-1);
count = zeros(size(centroids,1),n_markers+1);

if isequal(config.subtract_background,"true")
    back_vals = readmatrix(fullfile(config.output_directory,'background','back_values.csv'));
    back_vals = back_vals(~rm_idx,:);
else
    back_vals = zeros(1,n_markers-1);
end

% Clustering using Gaussian Mixtures
for i = 2:n_markers
    idx = 4+i;
    intensities = centroids(:,idx);

    % Subtract background
    if isequal(config.subtract_background,"true")
            fprintf('%s\t Subtracting background from centroids \n',datetime('now'))
            back_thresh = back_vals(:,i-1);
            %disp(std(back_thresh(:)))
            if i == 2
                intensities = intensities - back_thresh;
            end
    end
        
    % Apply z normalization
    if isequal(config.z_normalization,"true")% && ~isequal(config.stratify_structures,"true")
       fprintf('%s\t Applying z normalization \n',datetime('now'))
        intensities = (intensities - mean(intensities))/std(intensities);        
        if config.log_outliers ~= 0
            fprintf('%s\t Supressing outliers \n',datetime('now'))
            l = config.log_outliers;
            if isequal(config.stratify_structures,"false")
                intensities(intensities>l) = l + log10(intensities(intensities>l)-l+1);
                intensities(intensities<-l) = -l - log10(abs(intensities(intensities<-l))-l+1);
            end
        end
    end
    
    % If not splitting markers, continue for loop
    if ~isequal(config.split_markers,"true")
        values{i-1} = intensities;
        if i ~= n_markers
            continue
        end
        fprintf('%s\t Running GMM all markers with %d clusters \n',datetime('now'),n_clusters(1))
    else
        values = intensities;
        fprintf('%s\t Running GMM on marker %s with %d clusters \n',datetime('now'),markers(i),n_clusters(i-1))
    end
        
    % Perform GMM with all mixtures in 1 fitting
    if ~isequal(config.stratify_structures,"true")
        % Cluster all cells together
        [p{i-1}, gm{i-1}] = cluster_together(values,n_clusters(i-1),config);
    else
        % Cluster by structure
        structures = bin_annotation_structures(annotations);
        [p{i-1}, gm{i-1}] = cluster_by_structure(values,structures,layers,n_clusters(i-1),config);
    end
    
    % Threshold by confidence of NOT being in background cluster
    if config.n_clusters(i-1) > 2
        idx = p{i-1}(:,3)>0.999;
        %disp(sum(idx))
        p{i-1}(idx,3) = 0;
        p{i-1} = [p{i-1}(:,1), sum(p{i-1}(:,2:end),2)];
    end
    
    refine_layers = "true";
    if isequal(refine_layers,"true")
        con = repmat(confidence(i-1),size(p{i-1}(:,2)));
        if i == 2
            idx = ismember(layers,1:3);
        else
            idx = ismember(layers,4:7);
        end
        con(idx,:) = con(idx,:) + 0.4;
        count(:,i) = p{i-1}(:,2)>con; 
    else
        count(:,i) = p{i-1}(:,2)>confidence(i-1);        
    end
    
    % Finalize counts
    % count = finalize_counts(p,config.confidence);
end        

if n_markers > 2 && isequal(config.split_markers,"true")
    % Count cells positive for all markers and add to last column
    count(:,end) = all(count(:,2:end-1),2);
    count(:,2:end-1) = count(:,2:end-1) - count(:,end);
end

% Calculate all negative cells
count(:,1) = ~any(count(:,2:end),2);

% Save cell type counts
[r,c] = find(count);
v1(r) = c;

if ~any(cellfun(@(s) isempty(s),p))
    [v1,p] = divide_doubles_by_layer(v1,p,config,annotations);
end

ct(~rm_idx) = v1;
ct(rm_idx) = 0;

if isequal(remove_l1,"true")
    % Calculate posteriors
    ct(l1_idx) = 1;
end


% Calculate sums
max_ct = unique(ct(ct>0));
sums = zeros(1,length(max_ct));
for i = 1:length(max_ct)
    sums(i) = sum(ct==max_ct(i));
end
pct = sums/sum(sums(:));
disp(pct)

end


function [p,gm] = cluster_together(values, n_clusters, config)
%--------------------------------------------------------------------------
% Cluster on all structures
%--------------------------------------------------------------------------

% Get intial mixing parameters
init_struct = 'plus';

% Perform fitting
try
    gm = fitgmdist(values,n_clusters,'Start',init_struct,'Replicates',1,'Options',statset('MaxIter',200));
catch
    warning("Model fitting did not converge with initial parameters provided. Reinitializing....")
    gm = fitgmdist(values,n_clusters,'Options',statset('MaxIter',200));
end

[~,back_idx] = sort(gm.mu);
    
% Calculate posteriors
p = posterior(gm,values);
p = p(:,back_idx);

end


function [p,gm] = cluster_by_structure(val_save, structures,layers, n_clusters, config)
%--------------------------------------------------------------------------
% Stratify clustering by structure.
%--------------------------------------------------------------------------

p = zeros(size(val_save,1),n_clusters);
u_struct = unique(structures);

% Get intial mixing parameters
%init_struct = generate_intial_conditions(val_save,config.mix_proportions,config.mix_markers);
init_struct = 'plus';

% GMM fitting on each structure
for i = 1:length(u_struct)
    % Skip background
    if u_struct(i) == 0
        continue
    end
    fprintf('%s\t Running GMM on all markers with %d components on structure %d\n',...
        datetime('now'),n_clusters(1),i)
    
    idx = structures == u_struct(i);
    vals_sub = val_save(idx,:);    
    
    %a = std(vals_sub(layers(idx) == 1))/std(vals_sub(layers(idx) == 5))
    
    % Renormalize z for this structure
    if isequal(config.z_normalization,"true")
        vals_sub = (vals_sub - mean(vals_sub,1))./std(vals_sub,[],1);
        if config.log_outliers ~= 0
            l = config.log_outliers;
            vals_sub(vals_sub>l) = l;% + log10(vals_sub(vals_sub>l)-l+1);
            vals_sub(vals_sub<-l) = -l;% - log10(abs(vals_sub(vals_sub<-l))-l+1);
        end
    end
    
    % Perform fitting
    try
        gm = fitgmdist(vals_sub,n_clusters(1),'Start',init_struct,'Replicates',20,'Options',statset('MaxIter',400));
    catch
        warning("Model fitting did not converge with initial parameters provided. Reinitializing....")
        gm = fitgmdist(vals_sub,n_clusters(1),'Options',statset('MaxIter',400));
    end
    
    [~,back_idx] = sort(gm.mu);

    % Calculate posteriors
    p_sub = posterior(gm,vals_sub);
    p_sub = p_sub(:,back_idx);
    disp(sum(p_sub(:,1)>0.5)/size(p_sub,1))
    
    p(idx,:) = p_sub;
end

end


function [v1,p] = divide_doubles_by_layer(v1,p,config,annotations)

l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer1.csv'));
lu_idx = ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer2_3.csv'));
lu_idx = lu_idx | ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer4.csv'));
lu_idx = lu_idx | ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer5.csv'));
ll_idx = ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer6a.csv'));
ll_idx = ll_idx | ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer6b.csv'));
ll_idx = ll_idx | ismember(annotations,l_table.index);

%p1 = p{1}(:,1) < 0.1;
%p2 = p{2}(:,1) < 0.1;

%p1 = p1 & p{1}(:,1) < p{2}(:,1);
%p2 = p2 & p{1}(:,1) > p{2}(:,1);

p1 = p{1}(:,1).*ll_idx < p{2}(:,1).*ll_idx;
p2 = p{1}(:,1).*lu_idx < p{2}(:,1).*lu_idx;

v1(v1 == 4 & p1') = 2;
v1(v1 == 4 & p2') = 3;

end

function layers = get_structure_layers(annotations,config)

layers = zeros(size(annotations));

l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer1.csv'));
layers = layers + ismember(annotations,l_table.index);
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer2_3.csv'));
layers = layers + ismember(annotations,l_table.index)*2;
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer4.csv'));
layers = layers + ismember(annotations,l_table.index)*3;
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer5.csv'));
layers = layers + ismember(annotations,l_table.index)*4;
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer6a.csv'));
layers = layers + ismember(annotations,l_table.index)*5;
l_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer6b.csv'));
layers = layers + ismember(annotations,l_table.index)*6;

end
