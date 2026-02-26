function create_final_structure(config,path_table)
% Merge key analysis results for evaluation
% Sample ID 
% Most recent classes or centroids
% Most recent registration parameters
% Annotation mask
% Config structure from NM_analyze
% Save to output directory as .mat file

fprintf("%s\t Saving to results structure\n",datetime('now'))
if ~isfolder(config.output_directory)
    error("Could not locate output directory")
else
    var_dir = fullfile(config.output_directory,'variables');    
end

% Load previous results
fname = fullfile(config.output_directory, strcat(config.sample_id,'_results.mat'));
if isfile(fname)
    results = load(fname);
    if ~isequal(results.sample_id,config.sample_id)
        error("Sample id in the pre-loaded results structure does not match "+...
            "the sample id in the input configuration structure")
    end
else
    results = struct;
    results.sample_id = config.sample_id;
end

% Add image size and resolution at top level
[nrows,ncols] = size(read_img(path_table));
results.image_size = [nrows, ncols, length(unique(path_table.z))];
results.resolution = config.resolution{1};
results.orientation = config.orientation;

% Check registration parameters
if isfile(fullfile(var_dir,'reg_params.mat'))
    load(fullfile(var_dir,'reg_params.mat'),'reg_params')
    results.reg_params = reg_params;
end
    
% Check annotation mask
mfile = dir(fullfile(var_dir,'*_mask.mat'));
if isfile(fullfile(var_dir,mfile.name))
    load(fullfile(var_dir,mfile.name),'I_mask')
    results.I_mask = I_mask;
end

% Check for classes first, then centroids
%cfile = most_recent(dir(fullfile(config.output_directory,'*_classes*.csv')));
cfile = fullfile(var_dir,'classes.mat');
cfile2 = fullfile(var_dir,'centroids.mat');
if isfile(cfile)
    load(cfile,'classes')
    results.centroids = classes.centroids;
    results.annotations = classes.annotations;
    results.classes = classes.classes;
elseif isfile(cfile2)
    load(cfile2,'centroids')
    results.centroids = centroids.coordinates;
    results.annotations = centroids.annotations;
    if isfield(results,'classes')
        results = rmfield(results,'classes');
    end
end

% Save most recent config file
if isfile(fullfile(var_dir,'NM_variables.mat'))
    config = load(fullfile(var_dir,'NM_variables.mat'));
    results.config = config;
end

% Write structure
save(fname,'-struct','results')

end


function output = most_recent(input)
    
% Take most recent
if length(input) > 1
    idx = arrayfun(@(s) str2double(regexp(s.name(end-6:end-4),'\d*','match')),input);
    [~,idx] = max(idx);
    output = input(idx);
else
    output = input;
end

end