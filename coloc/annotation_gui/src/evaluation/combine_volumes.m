function df_results = combine_volumes(config,vol_results)
%--------------------------------------------------------------------------
% Measure structure volumes and calculate cell densities if counts are 
% present. Append to full multi-sample dataset for all annotated
% structures.
%--------------------------------------------------------------------------

% Read annotation indexes
df_temp = readtable(config.template_file);
df_temp = df_temp(df_temp.index>0,:);
annotation_indexes = df_temp.index;

% Read results structures
df_volumes = zeros(length(annotation_indexes),length(config.results_path));
for i = 1:length(config.results_path)
    var_names = who('-file',config.results_path(i));

    % Check for count results
    if ~any(ismember(var_names,'I_mask'))
        fprintf("%s\t No mask information found for sample %s. Skipping...\n",...
            datetime('now'),config.samples(i))
        continue
    end
    load(config.results_path(i),'summary')
    if ~ismember('summary',var_names) || ~isfield(summary,'volumes')
         summary.volumes = measure_structure_volumes(config.results_path(i));
         save(config.results_path(i),'-append','summary')
    end
    df_volumes(summary.volumes(:,1),i) = summary.volumes(:,2);
end

% Sum according to structure level order
ids = df_temp.id;
path = df_temp.structure_id_path;
df_new = zeros(size(df_volumes));
for i = 1:length(ids)
    idx = cellfun(@(s) contains(s,string("/"+ids(i)+"/")), path);
    df_new(i,:) = sum(df_volumes(idx,:),1);
end
df_volumes = df_new;

% Create header name
df_header = config.samples + "_" + "Volume";

% Convert to table
volumes_table = array2table(df_volumes,'VariableNames',df_header);

% Concatenate volumes to annotations
df_results = horzcat(df_temp(:,[1,end-1,end]), volumes_table);

% Write new results file
writetable(df_results,vol_results)

end