function df_results = combine_cortexes(config,cortex_results)
%--------------------------------------------------------------------------
% Measure structure volumes and calculate cell densities if counts are 
% present. Append to full multi-sample dataset for all annotated
% structures.
%--------------------------------------------------------------------------

% Read annotation indexes
ctx1 = readtable(fullfile(config.home_path,'annotations','custom_annotations','cortex_17regions.xls'));
ctx2 = readtable(fullfile(config.home_path,'annotations','custom_annotations','harris_cortical_groupings.xls'));
df_temp = [ctx1;ctx2];
df_temp = unique(df_temp,'rows');
annotation_indexes = df_temp.index;

% Read results structures
n_samples = length(config.results_path);
df_cortex = zeros(length(annotation_indexes),n_samples*3);
for i = 1:n_samples
    load(config.results_path(i),'cortex')
    df_cortex(:,i) = cortex.volume;
    df_cortex(:,i+n_samples) = cortex.sa;
    df_cortex(:,i+n_samples*2) = cortex.th;
end

% Create header name
df_header = cat(2,config.samples + "_" + "volume",...
    config.samples + "_" + "surfacearea",...
    config.samples + "_" + "thickness");

% Convert to table
volumes_table = array2table(df_cortex,'VariableNames',df_header);

% Concatenate volumes to annotations
df_results = horzcat(df_temp(:,[1,end-1,end]), volumes_table);

% Write new results file
writetable(df_results,cortex_results)

end