function save_registered_images(config,reg_params)

% Get reference and moving images
hemisphere = config.hemisphere;
direction = config.registration_direction;
params = reg_params.(direction);
if isequal(direction,"atlas_to_image")
    mov = "atlas"; ref = "image";
elseif isequal(direction,"image_to_atlas")
    mov = "image"; ref = "atlas";
elseif isequal(direction,"mri_to_atlas")
    mov = "mri"; ref = "atlas";
elseif isequal(direction,"atlas_to_mri")
    mov = "atlas"; ref = "mri";
elseif isequal(direction,"image_to_mri")
    mov = "image"; ref = "mri";
elseif isequal(direction,"mri_to_image")
    mov = "mri"; ref = "image";
else
    error("Unknown registration direction specified")
end

% Get number of images to transform 
nmov = length(params.mov_img_path);
nref = length(params.ref_img_path);

% Make registered folder
reg_dir = fullfile(config.output_directory,'registered');
if ~isfolder(reg_dir)
    mkdir(reg_dir)
end

% Copy reference image to registered directory
% Reference image remain in native orientation, resolution
fprintf('%s\t Moving reference images\n',datetime('now'))
for i = 1:nref
    fname = fullfile(reg_dir,...
        sprintf("%s_REF_%s_%d.nii",config.sample_id,...
        params.ref_channels(i),params.ref_res));
    img = read_img(params.ref_img_path(i));
    
    % Apply pre-alignment if they exist
    if i>1 && isfield(params,'pre_ref_tforms')
        if i == 2
            fprintf('%s\t Applying pre-alignments to reference images\n',datetime('now'))
        end
        img = apply_prealignment(img,params.pre_ref_tforms{i-1}, ref,...
            params.ref_res, params.ref_orientation, config.hemisphere);   
    else
        img = convert_nii_16(img, ref, true);
        [nrows,ncols,nslices] = size(img);
    end
    niftiwrite(uint16(img),fname)
end

% Load moving image
fprintf('%s\t Applying transformations to moving images\n',datetime('now'))
for i = 1:nmov
    fname = fullfile(reg_dir,...
        sprintf("%s_MOV_%s_%d.nii",config.sample_id,...
        params.mov_channels(i),params.mov_res));
    img = read_img(params.mov_img_path(i));
    
    % Apply pre-alignment if they exist
    if i>1 && isfield(params,'pre_mov_tforms')
        if i == 2
            fprintf('%s\t Applying pre-alignments to moving images\n',datetime('now'))
        end
        img = apply_prealignment(img,params.pre_mov_tforms{i-1}, mov,...
            params.mov_res, params.mov_orientation, config.hemisphere);
    else
        %img = convert_nii_16(img, mov, hemisphere, true);
    end
    
    % Apply transformation
    img = standardize_nii(img, params.mov_res, params.mov_orientation, config.hemisphere, false, ...
        25, params.ref_orientation, config.hemisphere);
    img = transformix(double(img), params, [1,1,1], []);

    % Permute and resize to match reference
    %img = permute_orientation(img,'ail',char(params.ref_orientation));
    img = imresize3(img,[nrows,ncols,nslices]);
    
    niftiwrite(uint16(img),fname)
end

end


function img = convert_nii_16(img, img_type, hemisphere, convert)
% Convert nifti as single/double and convert to 16 bit uint

if nargin<4
    convert = false;
end

% Convert to double and adjust orientation
if isequal(class(img),'double') || isequal(class(img),'single')
    img = imrotate(img,90);
    img = flip(img,1);
else
    img = double(img); 
end

% Transform atlas_img to match sample orientation
if isequal(img_type,"atlas")
    if isequal(hemisphere, "right")
        img = flip(img,1);
    elseif isequal(hemisphere,"both")
        atlas_img2 = flip(img,3);
        img = cat(3,img,atlas_img2);
    elseif isequal(hemisphere,"left") 
    elseif ~isequal(hemisphere,"none")
        error("Unrecognized sample hemisphere value selected")
    end
end

% Convert to uint 16
if convert
    % Rescale intensities
    min_int = min(img,[],'all');
    max_int = max(img,[],'all');
    img = (img - min_int)/(max_int-min_int);
    img = uint16(img*65535);
end

end


function img = apply_prealignment(img, tforms, resolution, orientation, hemisphere)
% Apply pre-alignments 
% Images need to be converted to 25 um/voxel and permuteed to 'ail'
% orientation to maintain consistency with register_to_atlas

% Standardize image
[nrows, ncols, nslices] = size(img);

img = standardize_nii(img, resolution, orientation, hemisphere, false);
img = double(img);

% Apply transformation
img = transformix(img, tforms, [1,1,1], []);

% Resize and permute to original
img = permute_orientation(img,'ail',char(orientation));
img = imresize3(img,[nrows, ncols, nslices]);
img = uint16(img);

end