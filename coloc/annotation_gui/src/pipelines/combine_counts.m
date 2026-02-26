function df_results = combine_counts(config, count_results)
%--------------------------------------------------------------------------
% Measure cell count results and append to full multi-sample dataset for
% all annotated structures
%--------------------------------------------------------------------------

if isequal(config.use_classes,"true") && all(config.s_fields(:,3))
    classes = config.keep_classes;
    if isempty(classes)
        error("Please specify which classes to count as 'keep_classes'")
    else
        types_ind = config.class_names(classes);
    end
    using_classes = true;
else
    types_ind = "nuclei";
    classes = 1;
    using_classes = false;
end

% Read template or check for previous results to append to
df_temp = readtable(config.template_file);
df_temp = df_temp(df_temp.index>0,:);

if using_classes && isequal(config.sum_all_classes,"true")
    types_all = ["all",types_ind,config.custom_class];
else
    types_all = [types_ind,config.custom_class];
end

% Sum cell counts for each sample
df_results = cell(1,length(config.results_path));
for n = 1:length(config.results_path)
    % Choose sample
    fprintf("%s\t Reading sample %s...\n",datetime('now'),config.samples(n))
    sum_path = config.results_path(n);
    obj = matfile(sum_path);
    var_names = who(obj);

    % Check for count results
    if ~any(ismember(var_names,'centroids'))
        fprintf("%s\t No centroid information found for sample %s. Skipping...\n",...
            datetime('now'),config.samples(n))
        continue
    end
    
    % Read annotations
    if any(ismember(var_names,'annotations'))
        annotations = obj.annotations;
        annotations = annotations(annotations>0);
    else
        annotations = ones(size(obj.centroids,1),1);
    end
    
    % Read classes
    if using_classes
        ct = obj.classes;
        ct = ct(annotations>0);
        ct = ct(ismember(ct,config.keep_classes));
        annotations = annotations(ismember(ct,config.keep_classes));
        
        % Make sure number of cell-types present equals the number of config.markers
        assert(length(unique(ct)) == length(types_ind),"Number of specified cell types (%d) "+...
            "does not match unique cell type indexes (%d) in sample file.\n",length(unique(ct)),...
            length(types_ind))
    else
        ct = ones(size(annotations,1),1);
    end
    
    % Create count matrix
    counts = zeros(height(df_temp),length(types_ind));

    % Apply thresholds and count cells in each structure for each channel
    % First channel cell intensities start at column 5
    u_idx = unique(annotations);
    for i = 1:length(u_idx)
        df_sub = ct(annotations == u_idx(i));
        if using_classes
            for j = 1:length(classes)
                counts(u_idx(i),j) = sum(df_sub == classes(j));
            end
        else
            counts(u_idx(i)) = length(df_sub);
        end
    end

    % Sum according to structure level order, except for background
    ids = df_temp.id;
    path = df_temp.structure_id_path;
    for i = 2:length(ids)
        idx = cellfun(@(s) contains(s,string("/"+ids(i)+"/")), path);
        counts(i,:) = sum(counts(idx,:),1);
    end
    
    % Sum all classes
    if using_classes && isequal(config.sum_all_classes,"true")
        total_counts = sum(counts,2);
    else
        total_counts = [];
    end
    
    % Get any custom classes
    if ~isempty(config.custom_class)
        custom = get_custom_class(counts,types_ind,config.custom_class);
    else
        custom = [];
    end
    
    % Combine 
    counts = cat(2,total_counts,counts,custom);
        
    % Create header name
    df_header = config.samples(n) + "_" + config.groups{n}(2) + "_" + types_all + "_Counts";

    % Convert to table
    df_results{n} = array2table(counts,'VariableNames',df_header);
    
    % Print cells counted
    if using_classes
        for j = 1:length(types_all)
            fprintf('%s\t Total %s count: %d\n',datetime('now'),types_all(j),counts(2,j))
        end
    end
    %fprintf('%s\t Total cell count: %d\n',datetime('now'),length(ct))
end

% Concatenate counts to results table
df_results = cat(2,df_results{:});
if isempty(df_results)
    return
end
idx = repmat(types_all,1,length(config.results_path));
[~,idx] = sort(idx);
df_results = df_results(:,idx);
df_results = horzcat(df_temp, df_results);
writetable(df_results,count_results)

end
