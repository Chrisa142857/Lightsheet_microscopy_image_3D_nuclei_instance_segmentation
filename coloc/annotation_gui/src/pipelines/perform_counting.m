function config = perform_counting(config, path_table)
%--------------------------------------------------------------------------
% % Detect cell/nuclei centroids.
%--------------------------------------------------------------------------
% Usage:
% config = perform_counting(config, path_table)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analysis configuration structure.
%
% path_table: Path table containing image information.
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
%--------------------------------------------------------------------------

% Check for centroids structure
path_centroids = fullfile(config.var_directory,'centroids.mat');
path_save = fullfile(config.output_directory, sprintf('%s_centroids.csv',config.sample_id));

if isequal(config.count_nuclei,"false")
    fprintf('%s\t No cell counting selected\n',datetime('now'))
    return
elseif isequal(config.count_nuclei,"true") && isfile(path_centroids)
    fprintf('%s\t Centroids already detected. Skipping cell counting and saving to results structure \n',...
        datetime('now'))
    results = load(config.res_name);
    results.centroids = load_var(config,'centroids','coordinates');
    results.annotations = load_var(config,'centroids','annotations');
    if isfield(results,'classes')
        results = rmfield(results,'classes');
    end
    save(config.res_name,'-struct','results')
    counts = measure_cell_counts(config);
    save_to_summary(config.res_name,counts,'counts')
    return
end
   
% Append results to csv while progessing down stack
if isequal(config.count_method,"hessian")
    % Detect nuclei centroids using hessian-based blob detector
    path_sub = path_table(path_table.markers == config.markers(1),:); 
    
    % Check for intensity thresholds
    if isempty(config.lowerThresh) || isempty(config.signalThresh) ||...
            isempty(config.upperThresh)
        markers = config.markers;
        config.markers = config.markers(1);
        config = check_for_thresholds(config,path_sub,true);
        config.markers = markers;
    end
    
    % Load annotation mask
    if isequal(config.use_annotation_mask,"true")
        config.mask_file = fullfile(config.output_directory,...
            'variables',strcat(config.sample_id,'_mask.mat'));
        load(config.mask_file,'I_mask')
    else
        I_mask = [];
    end
    
    % Run prediction
    predict_centroids_hessian(config,path_sub,path_save,I_mask);
    
elseif isequal(config.count_method,"3dunet")
    % Detect nuclei centroids using 3dunet (requires python + conda environment)
    config.path_save = path_save;
    
    % Subset reference channel containing nuclei
    config.img_list = path_table(path_table.channel_num == 1,:).file;
    if ispc
        config.img_list = cellfun(@(s) strrep(s,'\','/'),config.img_list,'UniformOutput',false);
    end
    config.img_directory = fileparts(path_table(path_table.channel_num ==1,:).file{1});
    predict_centroids_3dunet(config)
end

% Read centroids csv file and resave as MATLAB structure
centroids = readmatrix(path_save);
coordinates = centroids(:,1:3);
save(path_centroids,'-v7.3','coordinates')
if size(centroids,2)>3
    annotations = centroids(:,4);
    save(config.res_name,'annotations','-append')
    save(path_centroids,'annotations','-append')
end
%delete(path_save)
save(config.res_name,'centroids','-append')

% Save to results structure
counts = measure_cell_counts(config);
save_to_summary(config.res_name,counts,'counts')

end