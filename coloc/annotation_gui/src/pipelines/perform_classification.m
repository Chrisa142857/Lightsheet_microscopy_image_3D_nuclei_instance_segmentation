function config = perform_classification(config, path_table)
%--------------------------------------------------------------------------
% Classify cell-types for cell nuclei.
%--------------------------------------------------------------------------
% Usage:
% config = perform_classification(config, path_table)
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
path_classes = fullfile(config.var_directory,'classes.mat');
if isequal(config.classify_cells,"false")
    fprintf('%s\t No cell classification selected\n',datetime('now'))
    return
elseif isequal(config.classify_cells,"true") && isfile(path_classes)
    fprintf('%s\t Centroids already classified. Skipping cell classification and saving to results structure \n',...
        datetime('now'))
    load(path_classes,'centroids')
    load(path_classes,'annotations')
    load(path_classes,'classes')

    save(config.res_name,'-append','centroids')
    save(config.res_name,'-append','annotations')
    save(config.res_name,'-append','classes')
    counts = measure_cell_counts(config);
    save_to_summary(config.res_name,counts,'counts')
    return
end

% Subset markers to classify
if isempty(config.classify_channels)
    % Use only channels at the same resolution as nuclear channel
    c = 1:length(config.markers);
    config.classify_channels = c(cellfun(@(s) isequal(s,config.resolution{1}),config.resolution));
else
    
end
class_markers = config.markers(config.classify_channels);
path_table = path_table(ismember(path_table.markers,class_markers),:);

% Load centroids
path_centroids = fullfile(config.var_directory,'centroids.mat');
if ~isfile(path_centroids)
    error("Could not locate centroids file in the output directory");
else
    fprintf('%s\t Loading centroid list \n',datetime('now'))
    centroids = load(path_centroids);
end

% Check if all channel intensity have been measured
if ~isfield(centroids,'intensities') || isequal(config.remeasure_centroids,'true') ||...
        size(centroids.intensities,2) ~= length(class_markers)
    % No intensities measured
    centroids.intensities = remeasure_centroids(centroids.coordinates,path_table,config,class_markers);
    centroids.intensities = round(centroids.intensities);
    save(path_centroids,'-struct','centroids')
end

% Calculate automatic minimum threshold if unspecified
if isempty(config.min_intensity)
    if ~isfile(fullfile(config.output_directory,'variables','thresholds.mat'))
        [~,~,thresh] = measure_images(config, path_table, 1, true);
    else
        thresh = load_var(config,'thresholds');
        thresh = thresh.signalThresh(1);
    end
    config.min_intensity = thresh;
    fprintf('%s\t Using minimum centroid intensity: %.0f \n',datetime('now'),thresh*65535)
end

% Name class file
path_classes = fullfile(config.var_directory,'classes.mat');
if isequal(config.classify_method,'gmm')
    % Not supported anymore. Might add later

elseif isequal(config.classify_method,'svm')
    % Classify cell-types by trained SVM
    % Check for centroid patches    
    save_directory = fullfile(config.output_directory,'classifier');
    if ~isfolder(save_directory)
        mkdir(save_directory)
    end
    
    % Read classifications + patch feautre info
    itable = fullfile(config.output_directory,'classifier',...
        sprintf('%s_patch_info.csv',config.sample_id));
    ftable = fullfile(config.output_directory,'classifier',...
        sprintf('%s_patch_features.csv',config.sample_id));
    ctable = fullfile(config.output_directory,'classifier',...
        sprintf('%s_classifications.csv',config.sample_id));
    ftable_full = fullfile(config.output_directory,'classifier',...
        sprintf('%s_full_patch_features.csv',config.sample_id));
    
    % Create new patches for manual labeling or load a previous set
    if ~isfile(itable) || ~isfile(ftable)
        fprintf('%s\t Generating new centroid patches for training SVM model \n',...
            datetime('now'))
        get_centroid_patches(centroids,path_table,config);
    % elseif isequal(config.load_patches,"true")
    %     get_centroid_patches('load',path_table,config);
    end
    
    % Get full patch features
    if ~isfile(ftable_full)
        fprintf("%s\t Generating complete table of centroid patch features. "+....
            "This may take some time...\n",datetime('now'));
        stable = get_patch_features(centroids,path_table,config);
        writematrix(stable,ftable_full)
    else
        fprintf('%s\t Loading complete table of patch features \n',datetime('now'))
        stable = readmatrix(ftable_full);
        if size(stable,1) ~= size(centroids.coordinates,1)
           error("Number of patch feature table does not match the number of centroids loaded")
        end
    end
    
    % Check for all tables prior to running classification
    if isfile(itable) && isfile(ftable) && ~isfile(ctable)
        error("Could not locate annotated classifications in classifier directory. "+...
            "Generate annotations using generate_classifications(config)")
    else
        ftable = readtable(ftable);
        ctable = readtable(ctable);
        
        % Check size
        if size(ftable,1) ~= height(ctable)
            error("Size of annotated classifications do not match size of feature table. "+...
                "Re-annotate patches using current settings.")
        end
        % Check if any empty
        % if any(ctable{:,1} == 0)
        %     error("Loaded annotations contain 0 values. All patches must be annotated with a value 1-9.")
        % end
    end
    
    % Classify cells
    ct = classify_cells_svm(centroids, stable, config);
    
elseif isequal(config.classify_method,'threshold')
    % Use simple thresholding for predictions
    ct = classify_cells_threshold(centroids, config);

else
    error('%s\t Invalid classification method specified \n',datetime('now'))
end

% Construct new csv file for cell classes
cen_classes = cat(2,centroids.coordinates,centroids.annotations,ct);

% Remove outliers
if isempty(config.keep_classes)
    config.keep_classes = unique(ct(ct>0)');
end
keep_idx = ismember(ct,config.keep_classes);
cen_classes = cen_classes(keep_idx,:);
centroids.intensities = centroids.intensities(keep_idx,:);

% Print results
fprintf('\t Total centroids retained: %d\n', size(cen_classes,1))
u_classes = unique(cen_classes(:,5));
for i = 1:length(u_classes)
    idx = cen_classes(:,5) == u_classes(i);
    count = sum(idx);
    intensities = round(mean(centroids.intensities(idx,:),1));
    fprintf('\t Classified %d cells or %.3f of centroids as type %d.\n',...
        count,count/size(cen_classes,1),u_classes(i))
    fprintf('\t Average intensities: %s.\n',num2str(intensities))
end

% Save to structure
centroids = cen_classes(:,1:3);
save(path_classes,'centroids')
annotations = cen_classes(:,4);
save(path_classes,'-append','annotations')
classes = cen_classes(:,5);
save(path_classes,'-append','classes')
res_name = fullfile(config.output_directory,strcat(config.sample_id,'_results.mat'));
save(res_name, '-append','centroids')
save(res_name, '-append','annotations')
save(res_name, '-append','classes')
counts = measure_cell_counts(config);
save_to_summary(config.res_name,counts,'counts')

fprintf('%s\t Classification of sample %s completed! \n',...
        datetime('now'),config.sample_id)
end
