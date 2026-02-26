function counts = measure_cell_counts(config)
% Calculate number of cells in each structure index for each class for one
% sample. Input is a NM_analyze config structure.

% Start from results structure
res_name = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));
var_names = whos('-file',res_name);

% Check for centroid
if ~any(ismember({var_names.name},'centroids'))
    error("No cells have been counted for this sample")
else
    cen_info = var_names(ismember({var_names.name},'centroids'));
    ncells = cen_info.size(1);
end

% Check for annotations field. If not present, set class number to 1
if any(ismember({var_names.name},'annotations'))
    load(res_name,'annotations')
else
    annotations = ones(ncells,1);
end

% Check for classes field. If not present, set class number to 1
if any(ismember({var_names.name},'classes'))
    load(res_name,'classes')
else
    classes = ones(ncells,1);
end

% Sum counts
u_class = unique(classes);
u_idx = unique(annotations);
counts = zeros(length(u_idx),max(u_class));
for i = 1:length(u_idx)
    df_sub = classes(annotations == u_idx(i));
    for j = 1:length(u_class)
        counts(i,u_class(j)) = sum(df_sub == u_class(j));
    end
end
fprintf('%s\t Total cells counted: %d\n',datetime('now'),sum(counts(:)))
counts = cat(2,u_idx,counts);

end