function save_to_summary(filepath, data, data_type)
% Append summary counts to results structure without overwriting subfields

if isstruct(filepath)
    filepath = fullfile(filepath.output_directory,strcat(filepath.sample_id,'_results.mat'));
end

var_names = who('-file',filepath);
if ismember('summary',var_names)
    load(filepath,'summary')
end
summary.(data_type) = data;
save(filepath,'-append','summary')

end