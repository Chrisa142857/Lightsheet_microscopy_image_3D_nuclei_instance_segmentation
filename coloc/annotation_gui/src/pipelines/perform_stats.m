function perform_stats(config, multi)
%--------------------------------------------------------------------------
% Calculate measurements and statitics for one or multiple images.
%--------------------------------------------------------------------------
% Usage:
% perform_stats(config, multi)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Evaluate configuration structure.
%
% multi: (logical) True if multiple samples.
%
%--------------------------------------------------------------------------

% Check if overwriting
overwrite = false;
if isequal(config.update, "true")
    overwrite = true;
end

% Check if measuring cortex
measure_cortex = false;
if isequal(config.measure_cortex, 'true')
    % Check for cortex annotation (
    r = load_results(config,'summary.volumes');
    if r(r(:,1) == 5, 2) == 0 
        warning("Cortex annotation (index=5) was not found in the sample results structure. " +...
            "Skipping cortical measurements.")
        pause(3)
    else
        measure_cortex = true;
    end
end

% Start combining regions statistics
config.tab_directory = fullfile(config.results_directory,config.prefix,'tables');
if ~isfolder(config.tab_directory)
    mkdir(config.tab_directory)
end

config.vox_directory = fullfile(config.results_directory,config.prefix,'voxels');
if ~isfolder(config.vox_directory)
    mkdir(config.vox_directory)
end

if measure_cortex
    config.flat_directory = fullfile(config.results_directory,config.prefix,'flatmaps');
    if ~isfolder(config.flat_directory)
        mkdir(config.flat_directory)
    end
end

n_samples = length(config.samples);

% Measure cell-types
count_results = fullfile(config.tab_directory,strcat(config.prefix,"_summary_counts.csv"));
if overwrite || ~isfile(count_results)
    fprintf('%s\t Quantifying cell counts for %d samples\n',datetime('now'),n_samples)    
    combine_counts(config, count_results);
else
    fprintf('%s\t Cell count results already quantified \n',datetime('now'))
end

% Measure volumes
vol_results = fullfile(config.tab_directory,strcat(config.prefix,"_summary_volumes.csv"));
if overwrite || ~isfile(vol_results)
    fprintf('%s\t Quantifying structure volumes for %d samples\n',datetime('now'),n_samples)
    combine_volumes(config,vol_results);
else
    fprintf('%s\t Structure volumes already quantified \n',datetime('now'))
end

% Measure densities
dense_results = fullfile(config.tab_directory,strcat(config.prefix,"_summary_densities.csv"));
if overwrite || ~isfile(dense_results)
    fprintf('%s\t Quantifying cell densities for %d samples\n',datetime('now'),n_samples)
    combine_densities(config,count_results,vol_results,dense_results);
else
    fprintf('%s\t Cell densities already quantified \n',datetime('now'))        
end

% Measure cortex 
cortex_results = fullfile(config.tab_directory,strcat(config.prefix,"_summary_ctxVolSATH.csv"));
if measure_cortex
    if overwrite || ~isfile(cortex_results)
        fprintf('%s\t Measuring cortical structure for %d samples\n',datetime('now'),n_samples)
        ctx_idx = arrayfun(@(s) {who('-file',s)},config.results_path);
        ctx_idx = cellfun(@(s) any(ismember(s,{'cortex'})),ctx_idx);
        if ~all(ctx_idx)
            measure_cortical_sa_th(config.results_path(~ctx_idx));
        end
        combine_cortexes(config,cortex_results);
    else
        fprintf('%s\t Cortical structures already quantified. Loading measurments  \n',datetime('now'))        
        
    end
end

% Read counts and volume data directly from csv file
df_results = cell(1,3);
if isfile(count_results)
    fprintf('%s\t Loading cell counts \n',datetime('now'))    
    df_results{1} = readtable(count_results,'PreserveVariableNames',true);
else
    %error('Could not locate %s in results directory',count_results)
end
if isfile(vol_results)
    fprintf('%s\t Loading structure volumes \n',datetime('now'))    
    df_results{2} = readtable(vol_results);
else
    %error('Could not locate %s in results directory',vol_results)
end
if measure_cortex
    if isfile(cortex_results)
        fprintf('%s\t Loading cortex measurements \n',datetime('now'))    
        df_results{3} = readtable(cortex_results);
    else
        %error('Could not locate %s in results directory',cortex_results)
    end
end

% Statistics run on mutliple samples
if ~multi || isempty(config.groups)
    fprintf('%s\t Only 1 sample so skipping statistics...\n',datetime('now'))
    return
end

% Get sample info from 
c =  cellfun(@(s) strsplit(s,'_'),df_results{1}.Properties.VariableNames(10:end),'UniformOutput',false);        
c = cat(1,c{:});

% Edit this, samples and groups should be read directly from column
% names
idx = ismember(config.samples,string(unique(c(:,1))))';
config.samples = config.samples(idx);
config.groups = config.groups(idx);
config.markers = string(unique(c(:,3),'stable'));

% Get sub-group
groups = cat(1,config.groups{:});
n_groups = size(groups,2);
if n_groups>1
    sub_groups = unique(groups(:,2));
    for i = 1:length(sub_groups)
        fprintf('%s\t Found %d samples in sub-group %s\n',datetime('now'),...
            sum(groups(:,2) == sub_groups(i)),sub_groups(i))
    end
end

% Get additional categorical covariates
if n_groups>2
    for i = 3:n_groups
        fprintf('%s\t Found %d categorical group with %d unique values \n',datetime('now'),...
            i-2, length(unique(groups(:,i))))
    end
end

% Calculate region statistics
config.groups = config.groups';
config.samples = config.samples';
calc_table_stats(df_results, config);

% Calculate voxel image
%calc_voxel_stats(config)

% Calculate flatmap stats
if measure_cortex
    calc_flatmap_stats(config)
end

end
