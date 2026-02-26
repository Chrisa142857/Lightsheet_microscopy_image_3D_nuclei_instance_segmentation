function [samples,groups,res_path,s_fields] = munge_results(var_path)
%--------------------------------------------------------------------------
% Get sample, group, and variable information from results path
%--------------------------------------------------------------------------
% Usage:
% [samples,groups,centroids_directory,s_fields] = munge_results(var_path)
%
%--------------------------------------------------------------------------
% Inputs:
% var_path: (string) Path to results structures.
%
%--------------------------------------------------------------------------
% Outputs:
% samples: (cell) Sample ID(s) for n samples in path.
%
% groups: (cell) Group assignment(s) for each sample.
%
% res_path: (string) Full path to results structures.
%
% s_fields: (nx3 logical) Specifies volume (1), count (2), and class (3)
% data is present for n samples.
%
%--------------------------------------------------------------------------

var_path = string(var_path);
s_path = cell(1,length(var_path));
for i = 1:length(var_path)
    if endsWith(var_path,'.mat')
        % Direct path to structure
        s_path{i} = var_path;
    else
        s_path{i} = dir(fullfile(var_path,'*_results.mat'));
    end
end
s_path = cat(1,s_path{:});
n_files = length(s_path);

samples = strings(1,n_files);
groups = cell(1,n_files);
res_path = strings(1,n_files);
s_fields = false(n_files,3);
for i = 1:n_files
    res_path(i) = fullfile(var_path,s_path(i).name);
    f = whos('-file',res_path(i));
    
    load(res_path(i),'sample_id','group')
    samples(i) = string(sample_id);
    groups(i) = {group};
    
    if any(contains({f.name},'I_mask')); s_fields(i,1) = true; end
    if any(contains({f.name},'centroids')); s_fields(i,2) = true; end
    if any(contains({f.name},'classes')); s_fields(i,3) = true; end
end
    
end