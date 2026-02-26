function results = load_results(input, variable, sample, contains_field)
%--------------------------------------------------------------------------
% Load results summary for a given sample. 
%--------------------------------------------------------------------------
% Usage:
% results = load_results(input,variable,sample)
%
%--------------------------------------------------------------------------
% Inputs:
% input: String specifying sample in NM_samples or path to output
% directory. Also accepts configuration structure.
%
% variable: (char) Load just specific variable from results structure. 
% (default: [])
%
% sample: Optional string specifying sample if NM_evaluate structure
% provided.
%
% contains_field: Optional logical. Just check if the field exists without
% loading data.
%
%--------------------------------------------------------------------------
% Outputs:
% results: Results structure.
%
%--------------------------------------------------------------------------
if nargin<2
    variable = [];
end

if nargin<4
    contains_field = false;
end

if isstruct(input) && isfield(input, 'results_path') 
    if nargin > 2
        input.output_directory = input.results_path(input.samples == string(sample));  
    else
        input.output_directory = input.results_path(1);
    end
end
home_path = fileparts(which('NM_config'));

if isstruct(input)
    if isfield(input,'output_directory')
       % Read output directory from structure
       var_path = input.output_directory;
    else
        error("No output directory specified in the configuration structure.")
    end
else
    if ~contains(input,{'/','\'})
        % Get output directory for sample
        fid = fopen(fullfile(home_path,'templates','NM_samples.m'));
        c = textscan(fid,'%s');
        fclose(fid);
        
        samples = c{:}(find(cellfun(@(s) isequal(s,'case'),c{:}))+1);
        samples = cellfun(@(s) string(s(2:end-1)),samples);
        if ismember(sample,samples)
            fprintf('%s\t Evaluating single sample %s \n',datetime('now'),sample)
            [~, var_path] = NM_samples(sample, false);
        else
            error("Could not locate sample specified in NM_samples.")
        end
    else
        % Get input is output directory
        var_path = input;
    end
end

% Load structure
if endsWith(var_path,'.mat')
    results = load(var_path);
    if contains_field
        results = isfield(results, variable);
        return
    end
    if ~isempty(variable)
        results = load_sub_variable(results, variable);
    end
else
    files = dir(fullfile(var_path,"*_results.mat"));
    if length(files) == 1
        results = load(fullfile(files.folder,files.name));
        if contains_field
            results = isfield(results, variable);
            return
        end
        if ~isempty(variable)
            results = load_sub_variable(results, variable);
        end
    elseif length(files) > 1
        error("More than one results structure detected in the output directory")
    else
        error("Did not detect results structure in the output directory")
    end
end

end


function results = load_sub_variable(results, variable)

v = strsplit(variable,'.');
for i = 1:length(v)
    results = results.(v{i});
end

end