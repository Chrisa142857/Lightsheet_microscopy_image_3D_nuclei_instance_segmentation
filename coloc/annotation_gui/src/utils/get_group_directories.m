function [img_directory, output_directory, group_samples] = get_group_directories(sample,group_idx)

% Scan NM_samples for all samples
home_path = fileparts(which('NM_config.m'));
fid = fopen(fullfile(home_path,'templates','NM_samples.m'));
c = textscan(fid,'%s');
sample_idx = c{:}(find(cellfun(@(s) isequal(s,'case'),c{:}))+1);

% Get all groups
groups = cell(1,length(sample_idx));
for i = 1:length(sample_idx)
    [~,~,groups{i}] = NM_samples(sample_idx{i}(2:end-1), false);
end
fclose(fid);

% Find group for current sample
idx = contains(string(sample_idx),string(sample));
sample_groups = groups{idx};

% Subset any specific group
if nargin>1
    sample_groups = sample_groups(group_idx);
end

% Find other samples in the same group
% This uses AND logic so if multiple groups, other samples must belong to
% all groups
idx = cellfun(@(s) all(ismember(sample_groups,s)),groups);
group_samples = sample_idx(idx);
group_samples = cellfun(@(s) s(2:end-1),group_samples,'UniformOutput',false);
n_samples = length(group_samples);

% Now return input/output directories for these samples
img_directory = cell(1,n_samples);
output_directory = cell(1,n_samples);
for i = 1:n_samples
    [img_directory{i},output_directory{i}] = NM_samples(group_samples{i}, false);
end

end