function predict_centroids_3dunet(config)
%--------------------------------------------------------------------------
% Wrapper function to run nuclei centroid predictions via 3D-Unet. Requires
% installation of the necessary python packages located in environment.yml
% via conda. 
%--------------------------------------------------------------------------

% Use the default chunk sizes for the default 3D-Unet trained model
config.chunk_size = [112,112,32];          % Chunk size of unet model
config.chunk_overlap = [16,16,8];          % Overlap between chunks
config.trained_resolution = [1.21,1.21,4]; % Resolution at which the model was trained. Only required if resampling
config.resample_chunks = "false";          % Resample images to match model resolution. This process takes significantly longer

% Save matlab's config structure in the directory
home_path = fileparts(which('NM_config'));
save_path = fullfile(home_path,'src','analysis','3dunet','config.mat');

% Adjust resample resolution resolution
config.resample_resolution = repmat(config.resample_resolution,1,3);
config.resolution = config.resolution{1};
config.mask_file = fullfile(config.output_directory,'variables',strcat(config.sample_id,'_mask.mat'));

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

% Convert to chars and save
config = convertContainedStringsToChars(config);
save(save_path,'config','-v7.3','-nocompression')

% Set pythonpath
pythonpath = fullfile(home_path, 'src','analysis','3dunet');
setenv('PYTHONPATH', pythonpath)

% Run prediction
fprintf('%s\t Running 3D-Unet prediction on GPU %s \n',datetime('now'),config.gpu)

if isunix
    command = sprintf("source activate 3dunet-centroid; "+...
        "export PYTHONPATH=%s; "+...
        "python %s --mat %s",...
        pythonpath,fullfile(pythonpath,'nuclei','generate_chunks.py'),save_path);
else
    command = sprintf("conda activate 3dunet-centroid && "+...
        "SET PYTHONPATH=%s&& "+...
        "python %s --mat %s",...
        pythonpath,fullfile(pythonpath,'nuclei','generate_chunks.py'),save_path);
end

if ~isempty(config.gpu)
    command = sprintf("%s --g %s",command,config.gpu);
end
    
% Display and run command
disp(command)
system(command,'-echo');

end
