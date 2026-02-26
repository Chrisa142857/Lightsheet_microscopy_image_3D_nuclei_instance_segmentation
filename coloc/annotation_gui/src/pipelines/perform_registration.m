function config = perform_registration(config, path_table_nii)
%--------------------------------------------------------------------------
% Run image registration.
%--------------------------------------------------------------------------
% Usage:
% config = perform_registration(config, path_table_nii)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Analysis configuration structure.
%
% path_table: Path table containing image information for downsampled
% images.
%
%--------------------------------------------------------------------------
% Outputs:
% config: Updated configuration structure.
%
%--------------------------------------------------------------------------


home_path = fileparts(which('NM_config'));
reg_dir = fullfile(config.output_directory,'registered');
direction = config.registration_direction;
reg_file = fullfile(config.output_directory,'variables','reg_params.mat');
mask_var = fullfile(config.output_directory,'variables',strcat(config.sample_id,'_mask.mat')); 
reg_params = []; 
run_registration = true;

if isequal(config.register_images,"false") && isempty(config.annotation_file)
    % Check if I_mask exists
    if ~isfile(mask_var) && isequal(config.use_annotation_mask,"true")
        error("Analysis is configured to use annotations however none exist. "+...
            "Run registration to generate an annotation mask or specify custom annotations")
    end
    return
end

% Create registered directory
if ~isfolder(reg_dir) && isequal(config.save_registered_images,"true")
    mkdir(reg_dir)
end

% Get moving and reference directions
if isequal(config.registration_direction,"image_to_atlas")
    config.mov_direction = "image"; config.ref_direction = "atlas";
elseif isequal(config.registration_direction,"image_to_mri")
    config.mov_direction = "image"; config.ref_direction = "mri";
elseif isequal(config.registration_direction,"mri_to_atlas")
    config.mov_direction = "mri"; config.ref_direction = "atlas";
elseif isequal(config.registration_direction,"mri_to_image")
    config.mov_direction = "mri"; config.ref_direction = "image";
elseif isequal(config.registration_direction,"atlas_to_image")
    config.mov_direction = "atlas"; config.ref_direction = "image";
elseif isequal(config.registration_direction,"atlas_to_mri")
    config.mov_direction = "atlas"; config.ref_direction = "mri";
else
    error("Unrecognized registration direction specified.")
end

% Attempt to load registration parameters
if isfile(reg_file) 
    % Load previous parameters if not updating
    if run_registration && ~isequal(config.register_images,"update")
        fprintf('%s\t Loading previosuly calculated registration parameters \n',datetime('now'))
        load(reg_file,'reg_params')
        % Check if loaded registration parameters contain the direction
        % specified in config
        if isfield(reg_params,direction)
            fprintf('%s\t Parameters already exist for specified direction. Skipping registration\n',...
                datetime('now'))
            run_registration = false;
        end
    end
end

% Check atlas files
if run_registration && contains(config.registration_direction,"atlas")
    atlas_path = arrayfun(@(s) {char(fullfile(config.home_path,'data','atlas',s))},config.atlas_file);
    assert(all(isfile(atlas_path)), "Could not locate reference atlas file specified")
        
    % Check annotations for atlas dimensions
    if ismember(config.atlas_file,{'ara_nissl_25.nii','average_template.nii'})
        atlas_hemisphere = 'left';
        atlas_res = 25;
    else
        %[~,fname] = fileparts(config.atlas_file);
        %ann_path = arrayfun(@(s) {char(fullfile(config.home_path,'data','annotation_data',s))},...
        %    strcat(fname,'.mat'));
        a = load(fullfile('data','annotation_data',config.annotation_file),'hemisphere','resolution');
        atlas_hemisphere = a.hemisphere;
        atlas_res = a.resolution;
        config.mask_cerebellum_olfactory = "false";
    end
    
    % Resize
    %atlas_res = cellfun(@(s) repmat(str2double(regexp(s,'\d*','Match')),1,3),{config.atlas_file},'UniformOutput',false);

    % Rule: all atlas file must be at the same resolution
    %assert(all(atlas_res{1} == atlas_res{end}),"All loaded atlas file must be at the same resolution")
end

% Perform pairwise registration
if run_registration
    fprintf('%s\t Performing image registration \n',datetime('now'))

    % Get moving image paths, subset channels, save into config
    if isequal(config.mov_direction,"image")
        [~, resample_table] = path_to_table(config,'resampled');
        idx = ismember(resample_table.markers,config.markers(config.registration_channels));
        mov_img_path = resample_table(idx,:).file;
        assert(length(unique(resample_table(idx,:).y_res)) == 1,...
            "All resampled images for registration must be at the same resolution")
        config.mov_res = unqiue(resample_table(idx,:).y_res);
        config.mov_orientation = config.orientation;
        config.mov_channels = config.markers(config.registration_channels);
        config.mov_hemisphere = config.hemisphere;
        if isequal(config.registration_prealignment,"image") ||...
                isequal(config.registration_prealignment,"both")
            config.mov_prealign = true;
        else
            config.mov_prealign = false;
        end
                
    elseif isequal(config.mov_direction,"mri")
        mov_img_path = path_table_nii.file;
        config.mov_res = config.mri_resolution;
        config.mov_orientation = config.mri_orientation;
        config.mov_channels = config.mri_channels;
        config.mov_hemisphere = "both";
        if isequal(config.registration_prealignment,"mri") ||...
                isequal(config.registration_prealignment,"both")
            config.mov_prealign = true;
        else
            config.mov_prealign = false;
        end
        
    elseif isequal(config.mov_direction,"atlas")
        % Type of atlas
        t = cell(1,length(config.atlas_file));
        for i = 1:length(config.atlas_file)
            t{i} = table(fullfile(home_path,'data','atlas',config.atlas_file),...
                atlas_res,atlas_res,atlas_res,'VariableNames',...
                {'file','y_res','x_res','z_res'});
        end
        mov_img_path = {cat(1,t{:}).file};
        config.mov_res = atlas_res;
        config.mov_orientation = "ail";
        config.mov_channels = "atlas";
        config.mov_prealign = false;
        config.mov_hemisphere = atlas_hemisphere;
    end

    % Get reference image paths
    if isequal(config.ref_direction,"image")
        [~, resample_table] = path_to_table(config,'resampled');
        % Subset channels to register
        idx = ismember(resample_table.markers,config.markers(config.registration_channels));
        ref_img_path = resample_table(idx,:).file;
        assert(~isempty(ref_img_path), sprintf("No resampled images found for marker %s",...
            config.markers(config.registration_channels)));
        assert(length(unique(resample_table(idx,:).y_res)) == 1,...
            "All resampled images for registration must be at the same resolution")
        config.ref_res = unique(resample_table(idx,:).y_res);
        config.ref_orientation = config.orientation;
        config.ref_channels = config.markers(config.registration_channels);
        config.ref_hemisphere = config.hemisphere;
        if isequal(config.registration_prealignment,"image") ||...
                isequal(config.registration_prealignment,"both")
            config.ref_prealign = true;
        else
            config.ref_prealign = false;
        end
        
    elseif isequal(config.ref_direction,"mri")
        ref_img_path = path_table_nii.file;
        config.ref_res = config.mri_resolution;
        config.ref_orientation = config.mri_orientation;
        config.ref_channels = config.mri_channels;
        config.ref_hemisphere = "both";
        if isequal(config.registration_prealignment,"mri") ||...
                isequal(config.registration_prealignment,"both")
            config.ref_prealign = true;
        else
            config.ref_prealign = false;
        end
        
    elseif isequal(config.ref_direction,"atlas")
        % Type of atlas
        t = cell(1,length(config.atlas_file));
        for i = 1:length(config.atlas_file)
            t{i} = table(fullfile(home_path,'data','atlas',config.atlas_file),...
                atlas_res,atlas_res,atlas_res,'VariableNames',...
                {'file','y_res','x_res','z_res'});
        end
        ref_img_path = {cat(1,t{:}).file};
        config.ref_res = atlas_res;
        config.ref_orientation = "ail";
        config.ref_channels = "atlas";
        config.ref_prealign = false;
        config.ref_hemisphere = atlas_hemisphere;
    end

    % Calculate registration parameters
    % Moving image to reference image
    reg_params.(direction) = register_to_atlas(config, mov_img_path, ref_img_path);

    % Save registration %
    save(reg_file,'reg_params')
    save(config.res_name,'-append','reg_params')
    
    % Calculate inverse if specified
    %%%%%%%
    fprintf('%s\t Registration completed! \n',datetime('now'))
end

 % Save a copy of registered images if previous image does not exist
 if isequal(config.save_registered_images,"true") %&& run_registration
    fprintf('%s\t Transforming and saving registered images \n',datetime('now'))
    save_registered_images(config, reg_params)
 end

 % Registration complete. Now calculate mask if specified
 if isequal(config.use_annotation_mask, "true")
     get_mask_from_parameters(config, reg_params)
 end

end