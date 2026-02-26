function [config, path_table] = NM_config(stage, sample, run)
%--------------------------------------------------------------------------
% Generate configured parameters for running a pipeline.
%--------------------------------------------------------------------------
% Usage:  
% config = NM_config(stage, sample, run)
% 
%--------------------------------------------------------------------------
% Inputs:
% stage: 'process', 'analyze', 'evaluate'
%
% sample: (string) Sample or group identifier(s) in NM_samples.
%
% run: (logical) Whether to run respective pipeline. (default: false)
%
%--------------------------------------------------------------------------
% Outputs:
% config: Parameter configuration structure.
%
%--------------------------------------------------------------------------

if nargin<1
    addpath(genpath(pwd))
    return
end

if nargin<3
    run = false;
end
    
% Elastix paths
% Swicth from 'default' and specify complete path for custom location
elastix_path_bin = 'default';
elastix_path_lib = 'default';
conda_path = 'default';

% Make sure path is set
home_path = fileparts(which('NM_config'));
addpath(genpath(home_path))
cd(home_path)

% Save template variables
if isequal(stage, 'evaluate')
    [main_stage, results_directory] = save_vars(stage);
else
    main_stage = save_vars(stage);
end
tmp_path = fullfile(home_path,'data','tmp','NM_variables.mat');

% Load and append sample info
if nargin > 1 && ~isequal(main_stage,'evaluate')
    [img_directory, output_directory] = NM_samples(sample, true);
    load(tmp_path,'-mat')
    
elseif nargin > 1 && isequal(main_stage,'evaluate')
    fid = fopen(fullfile(home_path,'templates','NM_samples.m'));
    c = textscan(fid,'%s');
    fclose(fid);

    % Get samples in group
    if ~isfolder(sample)
        samples = c{:}(find(cellfun(@(s) isequal(s,'case'),c{:}))+1);
        samples = cellfun(@(s) string(s(2:end-1)),samples);
        if ismember(sample,samples)
            % Sample input
            fprintf('%s\t Evaluating single sample %s \n',datetime('now'),sample)
            [~, output_directory] = NM_samples(sample, false);
        else
            % Group input
            fprintf('%s\t Evaluating group %s \n',datetime('now'),sample)
            groups = cell(1,length(samples));
            for i = 1:length(samples)
                [~, output_directory(i), groups{i}] = NM_samples(samples(i),false);
            end
            idx = cellfun(@(s) any(contains(s,sample)),groups);
            output_directory = output_directory(idx);
        end
        
        disp(output_directory)
        [samples,groups,results_path,s_fields] = munge_results(output_directory);
        
        assert(~isempty(samples),"No samples found for input %s",sample)
        assert(length(samples) == length(output_directory),...
            "Missing results structures for some samples")
    else        
        % Directory input containing multiple results structures
        [samples,groups,results_path,s_fields] = munge_results(sample);
    end
    
    % Check parameters
    if size(unique(s_fields,'rows'),1) ~= 1
        idx = any(s_fields,1);
        for i = 1:size(samples,1)
            if samples(i,1) && idx(1)
                warning("Volume information missing for sample %s",samples{i})
            end
            if samples(i,2) && idx(2)
                warning("Cell count information missing for sample %s",samples{i})
            end
            if samples(i,3) && idx(3)
                warning("Cell class information missing for sample %s",samples{i})
            end
        end
    end
    
    % Define output directory
    if isempty(results_directory)
        results_directory = fullfile(home_path,'results');
    end

    % Check if 1 or more samples
    n_samples = length(samples);
    if n_samples == 0
        prefix = [];
        warning("No samples present to evaluate")
    elseif n_samples > 1
        prefix = groups{1}(1);
    else
        prefix = samples(1);
    end
    
    output_directory = results_directory;
    use_processed_images = "false";
    clear sample;
    save(tmp_path,'prefix','samples','s_fields','results_path','groups','results_directory','-mat','-append')
else
    error("Sample information is unspecified. Set 'sample' variable.")
end

% Check variable lengths for some variables
check_variable_lengths(main_stage)

% Update image directory if using processed or analyzed images
if ~isequal(use_processed_images,"false")
    process_directory = fullfile(output_directory,use_processed_images);
    if ~exist(process_directory,'dir')
        warning("Could not locate processed image directory %s.\n" + ...
            "If not running any processing steps, set use_processed_images "+...
            "to ""false""", process_directory)
    else
        save(tmp_path,'process_directory','-mat','-append')
    end
elseif ~isequal(main_stage,'evaluate')
    if ~isfolder(img_directory)
        error("Could not find image directory")
    end
end

% Add elastix to PATH
if isequal(elastix_path_bin,'default')
    elastix_path_bin = fullfile(home_path,'src','external','elastix','bin');
end

if isequal(elastix_path_lib,'default')
    elastix_path_lib = fullfile(home_path,'src','external','elastix','lib');
end
add_elastix_to_path(elastix_path_bin,elastix_path_lib)

% Add conda to PATH
PATH = getenv('PATH');
if isequal(conda_path,'default')
    add_conda_to_path;
elseif ~contains(PATH,'conda3')
    PATH = conda_path + ":" + PATH; 
    setenv('PATH',PATH);
end

% Check if MATLAB toolboxes exist
try
    gcp('nocreate');
catch
    warning("Could not load Parallel Computing Toolbox. It's recommended "+...
        "that this toolbox be installed to speed up analysis.")
end

% Reload config
config = load(tmp_path);
config = orderfields(config);

% Run
if run
    % Make an output directory
    if exist(output_directory,'dir') ~= 7
        mkdir(output_directory);
    end

    % Make a variables directory
    var_directory = fullfile(output_directory,'variables');
    if exist(var_directory,'dir') ~= 7
        mkdir(var_directory);
    end
    
    % Copy variables to output destination
    copyfile(tmp_path,var_directory)
    
    % Run pipeline
    switch stage
        case 'process'
            NM_process(config)
        case 'analyze'
            NM_analyze(config,'analyze')
        case 'evaluate'
            NM_evaluate(config)
        case 'stitch'
            NM_process(config,'stitch',true)
        case 'align'
           NM_process(config,'align',true)
        case 'intensity'
           NM_process(config,'intensity',true)
        case 'resample'
           NM_analyze(config,'resample')
        case 'register'
           NM_analyze(config,'register')
        case 'count'
           NM_analyze(config,'count')
        case 'classify'
           NM_analyze(config,'classify')
    end
end

if nargout == 2
    path_table = path_to_table(config);
end

end


function check_variable_lengths(stage)
% Check user input variable lengths to make sure they're the correct length
% and match number of markers in most cases

switch stage
    case 'process'
        % Variables to check
        variable_names = {'markers','ignore_markers','channel_num','single_sheet','blending_method',...
            'elastix_params','rescale_intensities','subtract_background','Gamma','smooth_img','smooth_sigma',...
            'DoG_img','DoG_minmax','DoG_factor','darkfield_intensity', 'resolution','z_initial',...
            'lowerThresh','upperThresh','signalThresh'};
        load(fullfile('data','tmp','NM_variables.mat'),variable_names{:});
        
        if exist('markers','var') ~= 1  || isempty(markers)
            error("Must provide unique marker names for channel");end
        if exist('ignore_markers','var') == 1  && ~isempty(ignore_markers)
            ig_idx = ismember(markers,ignore_markers);
            markers = markers(~ig_idx);   
            if exist('channel_num') == 1 && ~isempty(channel_num)
                channel_num = channel_num(~ig_idx);
            end
        end
        if exist('single_sheet','var') == 1 && length(single_sheet) == 1
            single_sheet = repmat(single_sheet,1,length(markers));end
        if exist('blending_method','var') == 1 && length(blending_method) == 1
            blending_method = repmat(blending_method,1,length(markers));end
        if exist('elastix_params','var') == 1 && length(elastix_params) == 1
            elastix_params = repmat(elastix_params,1,length(markers)-1);end
        if exist('rescale_intensities','var') == 1 && length(rescale_intensities) == 1
            rescale_intensities = repmat(rescale_intensities,1,length(markers));end
        if exist('subtract_background','var') == 1 && length(subtract_background) == 1
            subtract_background = repmat(subtract_background,1,length(markers));end
        if exist('Gamma','var') == 1 && length(Gamma) == 1 || isempty(Gamma)
            Gamma = ones(1,length(markers));end
        if exist('smooth_img','var') == 1 && length(smooth_img) == 1
            smooth_img = repmat(smooth_img,1,length(markers));end
        if exist('smooth_sigma','var') == 1 && isempty(smooth_sigma)
            smooth_sigma = ones(1,length(markers));end
        if exist('smooth_sigma','var') == 1 && length(smooth_sigma) == 1
            smooth_sigma = repmat(smooth_sigma,1,length(markers));end
        if exist('DoG_img','var') == 1 && length(DoG_img) == 1
            DoG_img = repmat(DoG_img,1,length(markers));end
        if exist('DoG_minmax','var') == 1 && isempty(DoG_minmax) == 1
            DoG_minmax = [0.8,1.5];end
        if exist('DoG_factor','var') == 1 && length(DoG_factor) == 1
            DoG_factor = repmat(DoG_factor,1,length(markers));end
        if exist('darkfield_intensity','var') == 1 && length(darkfield_intensity) == 1
            darkfield_intensity = repmat(darkfield_intensity,1,length(markers));end
        if exist('resolution','var') == 1
            if ~iscell(resolution)
                resolution = {resolution};
            end
            if length(resolution) == 1
                resolution = repmat(resolution,1,length(markers));
            end
            resolution = resolution(~ig_idx);
        end
        if exist('z_initial','var') == 1 
            if isempty(z_initial) 
                z_intial = zeros(1,length(markers));
            elseif length(z_initial) == 1
                z_initial = [0,repmat(z_initial,1,length(markers)-1)];
            end
        end
        if exist('lowerThresh','var') == 1 && ~isempty(lowerThresh)
            assert(length(lowerThresh) == length(markers),"Lower threshold values need to be "+...
                "specified for all markers or left empty")
            if isempty(lowerThresh) && ~isempty(upperThresh)
                upperThresh = zeros(0,1,length(markers));
            end
        end
        if exist('upperThresh','var') == 1
            if ~isempty(upperThresh)
                assert(length(upperThresh) == length(markers),"Upper threshold values need to be "+...
                    "specified for all markers or left empty")
            end
            if isempty(upperThresh) && ~isempty(lowerThresh)
                upperThresh = repmat(65535,1,length(markers));
            end
        end
        if exist('signalThresh','var') == 1
            if ~isempty(signalThresh)
                assert(length(signalThresh) == length(markers),"Signal threshold values need to be "+...
                    "specified for all markers or left empty")
            elseif ~isempty(lowerThresh) || ~isempty(upperThresh)
                error("signalThresh must specified if lowerThresh and/or upperThresh "+...
                    "is also specified")
            end
        end
    case 'analyze'
        % Variables to check
        variable_names = {'markers','resolution','lowerThresh','upperThresh','signalThresh',...
            'registration_direction'};
        load(fullfile('data','tmp','NM_variables.mat'),variable_names{:});
        
        if exist('markers','var') ~= 1  || isempty(markers)
            error("Must provide unique marker names for channel");end
        if exist('registration_direction','var') ~= 1
            if isempty(registration_direction)
                direction = "atlas_to_image";
            end
            assert(isequal(direction,"atlas_to_image") || isequal(direction,"image_to_atlas"),...
                "Invalid option for registration direction")
        end
        if exist('resolution','var') == 1
            if ~iscell(resolution) && length(resolution) == 3
                resolution = {resolution};
                resolution = repmat(resolution,1,length(markers));
            end
        end
        if exist('lowerThresh','var') == 1 && ~isempty(lowerThresh)
            assert(length(lowerThresh) == length(markers),"Lower threshold values need to be "+...
                "specified for all markers or left empty")
            if isempty(lowerThresh) && ~isempty(upperThresh)
                upperThresh = zeros(0,1,length(markers));
            end
        end
        if exist('upperThresh','var') == 1
            if ~isempty(upperThresh)
                assert(length(upperThresh) == length(markers),"Upper threshold values need to be "+...
                    "specified for all markers or left empty")
            end
            if isempty(upperThresh) && ~isempty(lowerThresh)
                upperThresh = repmat(65535,1,length(markers));
            end
        end
        if exist('signalThresh','var') == 1
            if ~isempty(signalThresh)
                assert(length(signalThresh) == length(markers),"Signal threshold values need to be "+...
                    "specified for all markers or left empty")
            elseif ~isempty(lowerThresh) || ~isempty(upperThresh)
                error("signalThresh must specified if lowerThresh and/or upperThresh "+...
                    "is also specified")
            end
        end
        
end

% Resave these variables
variable_names = who;
save(fullfile('data','tmp','NM_variables.mat'),variable_names{:},'-mat','-append')

end


function [main_stage, results_directory] = save_vars(stage)

% Load config variables
switch stage
    case {'process','stitch','align','intensity'}
        NMp_template
        main_stage = 'process';
    case {'analyze','resample','register','count','classify'}
        NMa_template
        main_stage = 'analyze';
    case {'evaluate','stats','visualize'}
        NMe_template
        main_stage = 'evaluate';
    otherwise
        error("Invalid stage input")
end

% Save config structure
save(fullfile('data','tmp','NM_variables.mat'),'-mat')

end
