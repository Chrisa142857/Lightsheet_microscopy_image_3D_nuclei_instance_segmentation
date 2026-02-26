function reg_params = register_to_atlas(config, mov_img_path, ref_img_path, num_points)
% Register images to the reference atlas using elastix via melastix
% wrapper. Note: the final registration parameters are stored in the
% reg_params variable. While it is simpler to just register the atlas to
% the image, you may, in some cases, get better accuracy by registering the
% image to the atlas and calculating the inverse parameters. 

% Defaults
high_prct = 99;                 % Max intensity adjustment percentile for images
Gamma = 0.9;                    % Gamma to apply for intensity adjustment
%inverse_params = "inverse";     % Location of elastix parameter for calculating the inverse

% Unpack config variables
params = config.registration_parameters;
output_directory = config.output_directory;
use_mask = config.mask_cerebellum_olfactory;
direction = config.registration_direction;
use_points = config.use_points;
atlas_file = config.atlas_file;
home_path = fileparts(which('NM_config.m'));
%cen_structure = config.prealign_annotation_index;
cen_structure = [];

% Unless testing, use all points
if nargin<4 || isempty(num_points)
    num_points = 'all';
end

% Remove tmp directories
cleanup_tmp(config)

% Location of parameters
if isempty(params)
    if isequal(use_points, "true")
        params = "points";
    else
        params = "default";
    end
end

% Type of atlas
if contains(atlas_file,'average')
    atlas_type = 'average';
else
    atlas_type = 'nissl';
end
    
% Convert to cell
if ~iscell(mov_img_path)
    mov_img_path = {mov_img_path};
end
fprintf('\t Reading moving and reference images\n'); 

% Load moving and reference images
mov_img_array = cell(1,length(mov_img_path));
ref_img_array = cell(1,length(ref_img_path));
for i = 1:length(mov_img_path)
    mov_img = read_img(mov_img_path{i});
    
    % Standardize and save into array
    mov_img_array{i} = standardize_nii(mov_img, config.mov_res,...
        config.mov_orientation, config.mov_hemisphere, false, 25, config.ref_orientation, ...
        config.ref_hemisphere, 'double');
end
for i = 1:length(ref_img_path)
    ref_img = read_img(ref_img_path{i});
    
    % Standardize and save into array
    ref_img_array{i} = standardize_nii(double(ref_img), config.ref_res,...
        config.ref_orientation, config.ref_hemisphere, false, 25, config.ref_orientation, ...
        config.ref_hemisphere, 'double');
end
size_mov = size(mov_img_array{1});
size_ref = size(ref_img_array{1});

% Adjust intensities
mov_img_array = cellfun(@(s) double(imadjustn(uint16(s),...
    [0,double(prctile(s(:),high_prct))/65535],[],Gamma)), mov_img_array,...
    'UniformOutput', false);

ref_img_array = cellfun(@(s) double(imadjustn(uint16(s),...
    [0,double(prctile(s(:),high_prct))/65535],[],Gamma)), ref_img_array,...
    'UniformOutput', false);
    
% Chain registration parameters. Parameter files are found in
% data/elastix_parameter_files/atlas_registration. 
% Update specific elastix parameters by editting these files.
% Paths to elastix parameter files
param_path = fullfile(home_path,'data','elastix_parameter_files','atlas_registration',params);
param_path = dir(param_path);
parameter_paths = cell(1,length(params));
if ~isempty(param_path)
    % Detect which transform is in the parameter file
    parameterSub = param_path(arrayfun(@(s) endsWith(s.name,'.txt'),param_path));
    for j = 1:length(parameterSub)
        file_path = fullfile(parameterSub(1).folder,parameterSub(j).name);
        text = textread(file_path,'%s','delimiter','\n');
        n = find(cellfun(@(s) contains(s,'(Transform '),text));
        if contains(text(n),'Translation') || contains(text(n),'Affine') || contains(text(n),'Euler')
            fprintf('\t Performing rigid registration\n');
            parameter_paths{1} = file_path;
        elseif contains(text(n),'BSpline')
            fprintf('\t Performing b-spline registration\n');
            parameter_paths{2} = file_path;
        end
    end
else
    error("Could not locate elastix parameter folder %s.",param_path)
end

% Load registration points
points = [];
if isequal(use_points,"true")
    fprintf('\t Loading points to guide registration\n');

    % Load points from BigWarpJ .csv file
    [mov_points,ref_points] = load_points_from_bdv(output_directory, atlas_type, mov_img);

    % Trim points if not using all
    if isequal(num_points,'all')
        num_points = size(mov_points,1);
    end
    points.mov_points = mov_points(1:num_points,:);
    points.ref_points = ref_points(1:num_points,:);
end

% Add mask for olfactory and cerebellum
mask = [];
if isequal(use_mask,"true") && contains(direction,"atlas")
    load(fullfile(home_path,'data','annotation_data','olf_cer.mat'),'bw_mask')
    mask = standardize_nii(single(~bw_mask), 25,config.ref_orientation,config.hemisphere, true,...
        25, config.ref_orientation, config.ref_hemisphere, 'double');
end

% If pre-aligning by structure index, load annotations and find centroid of
% the given structure. Then use centroid positions as input for points
% registration
cen_points = [];
if ~isempty(cen_structure)
    assert(length(cen_structure), "Can only center on 1 annotation index")
    assert(isequal(config.annotation_mapping, 'atlas'), "Can only support annotation "+...
        "centering when annotations are mapped to the atlas")

    % Get ref_img centroid
    cen_points.ref_points = size(ref_img_array{1})/2;

    % Load annotations. These should be in the .mat file
    av = load(fullfile(config.home_path, 'data', 'annotation_data', config.annotation_file));
    assert(ismember(cen_structure,av.annotationIndexes), "Annotation index is not in the annotation file. " +...
        "Make sure to choose an index at the bottom of the structure hierarchy.")
    A = av.annotationVolume;

    % Standardize and get centroid
    A = standardize_nii(A, av.resolution, av.orientation, av.hemisphere, true, 25, config.ref_orientation,...
                config.ref_hemisphere,'double');
    A = A == cen_structure;
    cen_points.mov_points = regionprops3(A, 'Centroid').Centroid([2,1,3]);
    cen_points.index = cen_structure;
end

% Perform pairwise registration
reg_params = elastix_registration(mov_img_array,ref_img_array,...
    parameter_paths, points, mask, config.output_directory,...
    direction, [config.mov_prealign, config.ref_prealign], cen_points);

% Create new variable or attach to existing
reg_params.mov_img_path = mov_img_path;
reg_params.mov_orientation = config.mov_orientation;
reg_params.mov_res = config.mov_res;
reg_params.mov_size = size_mov([2 1 3]);
reg_params.mov_channels = config.mov_channels;
reg_params.ref_img_path = ref_img_path;
reg_params.ref_orientation = config.ref_orientation;
reg_params.ref_res = config.ref_res;
reg_params.ref_size = size_ref([2 1 3]);
reg_params.ref_channels = config.ref_channels;


% Calculating inverse deprecated for now
% Return reg_params
return

% Calculate the inverse
if isequal(calc_inverse,"true")
    if isequal(direction,"atlas_to_image")
        direction = "image_to_atlas";
    elseif isequal(direction,"image_to_atlas")
        direction = "atlas_to_image";
    elseif isequal(direction,"image_to_mri")
        direction = "mri_to_image";
    elseif isequal(direction,"mri_to_image")
        direction = "image_to_mri";
    elseif isequal(direction,"mri_to_atlas")
        direction = "atlas_to_mri";
    elseif isequal(direction,"atlas_to_mri")
        direction = "mri_to_atlas";
    end
    
    fprintf('\t Getting inverse transformation parameters\n');
    
    if isequal(direction, "atlas_to_image")
        inv_direction = "image_to_atlas";
        reg_params.atlas_to_image = get_inverse_transform_from_atlas(config,...
            mov_img_array{1},reg_params,inv_direction,inverse_params);
        reg_params.atlas_to_image.TransformParameters{1}.Size = size_atlas([2 1 3]);
    else
        inv_direction = "atlas_to_image";
        reg_params.atlas_to_image = get_inverse_transform_from_atlas(config,...
            ref_img_array{1},reg_params,inv_direction,inverse_params);
        reg_params.atlas_to_image.TransformParameters{1}.Size = size_mov([2 1 3]);
    end
end

% Check results
%imshowpair(reg_img(:,:,120),atlas_img(:,:,120))

end


function [reg_params, reg_img, status] = elastix_registration(mov_img,ref_img,...
    parameter_paths,points,mask,outputDir,direction,prealign, cen_points)
% Single channel, pairwise registration
home_path = fileparts(which('NM_config.m'));

% Here mask is assumed to be only on the true atlas image
if isequal(direction,'atlas_to_image') || isequal(direction,'atlas_to_mri')
    atlas = "mov";
elseif isequal(direction,'image_to_atlas') || isequal(direction,'mri_to_atlas')
    atlas = "ref";
else
    atlas = "none";
end

% Check if using points
use_points = false;
if ~isempty(points)
    use_points = true;
    mov_points = points.mov_points;
    ref_points = points.ref_points;
end

% Pre-align by annotation index
pre_mov_annotation = false; pre_ref_annotation = false;
if ~isempty(cen_points) && isequal(atlas, "mov")
    fprintf('\t Pre-aligning reference images to match structure index %d\n', cen_points.index);    
    
    % Align images based on overlapping centroid positions using landmark registration
    pre_align_path = {fullfile(home_path,'data','elastix_parameter_files',...
        'atlas_registration','pre-align_annotation','ElastixParameterAffinePoints.txt')};

    % Create temporary directory for saving images
    outputDir = fullfile(outputDir,sprintf('tmp_reg_%d',randi(1E4)));
    if ~isfolder(outputDir)
        mkdir(outputDir)
    end
    
    [reg_params,reg_img,status]=elastix(mov_img,ref_img,outputDir,pre_align_path,...
        'fp',cen_points.ref_points,'mp',cen_points.mov_points,'threads',[]);

    pre_mov_annotation = true;
end

% If multiple moving channels, apply rigid registration to pre-align
pre_mov = false; pre_ref = false;
if length(mov_img)>1 && prealign(1)
    fprintf('\t Pre-aligning multiple moving image channels prior to registration\n');    
    [pre_mov_tforms, mov_img] = pre_align_rigid(mov_img,outputDir);
    pre_mov = true;
end

if length(ref_img)>1 && prealign(2)
    fprintf('\t Pre-aligning multiple reference image channels prior to registration\n');    
    [pre_ref_tforms, ref_img] = pre_align_rigid(ref_img,outputDir);
    pre_ref = true;
end

% Create temporary directory for saving images
outputDir = fullfile(outputDir,sprintf('tmp_reg_%d',randi(1E4)));
if ~isfolder(outputDir)
    mkdir(outputDir)
end

% Perform registration
if isempty(mask)
    % Register atlas to image with or without corresponding points
    if ~use_points
        [reg_params,reg_img,status] = elastix(mov_img,ref_img,...
            outputDir,parameter_paths,'threads',[]);
    else
        [reg_params,reg_img,status] = elastix(mov_img,ref_img,...
            outputDir,parameter_paths,'mp',points.mov_points,'fp',...
            points.ref_points,'threads',[]);
    end
else
    % Register atlas to image with or without corresponding points +
    % with a mask
    if ~use_points
        if isequal(atlas,"mov")
            [reg_params,reg_img,status]=elastix(mov_img,ref_img,...
                outputDir,parameter_paths,'mMask',mask,'threads',[]);
        elseif isequal(atlas,"ref")
            [reg_params,reg_img,status]=elastix(mov_img,ref_img,...
                outputDir,parameter_paths,'fMask',mask,'threads',[]);
        else
            reg_params = [];
            status = 1;
        end
    else
        if isequal(atlas,"mov")
            [reg_params,reg_img,status]=elastix(mov_img,ref_img,...
                outputDir,parameter_paths,'fp',ref_points,'mp',mov_points,...
                'mMask',mask,'threads',[]);
        elseif isequal(atlas,"ref")
            [reg_params,reg_img,status]=elastix(mov_img,ref_img,...
                outputDir,parameter_paths,'fp',ref_points,'mp',mov_points,...
                'fMask',mask,'threads',[]);
        else
            reg_params = [];
            status = 1;
        end
    end
end

% Remove transformed image from structure
reg_params.transformedImages = [];

% Remove temporary directory
rmdir(outputDir,'s')

% Check status and raise error if registration failed
if status ~= 0
    error("Errors occured during registration")
end

% Attach pre-alignment tforms if present
if pre_mov
    reg_params.pre_mov_tforms = pre_mov_tforms;
end
if pre_ref
    reg_params.pre_ref_tforms = pre_ref_tforms;
end

end


function [pre_tforms,img_array] = pre_align_rigid(img_array,outputDir)
% Rigid pre-alignment of images. Align multiple channels to each other. Or
% align images based on overlapping centroid positions using landmark
% registration


% Location of parameter files
home_path = fileparts(which('NM_config'));
pre_align_path = {fullfile(home_path,'data','elastix_parameter_files',...
    'atlas_registration','pre-align','ElastixParameterAffine.txt')};

% Create temporary directory for saving images
outputDir = fullfile(outputDir,sprintf('tmp_reg_%d',randi(1E4)));
if ~isfolder(outputDir)
    mkdir(outputDir)
end

% Run alignment
pre_tforms = cell(1,length(img_array)-1);
for i = 2:length(img_array)
    [pre_tforms{i-1},img_array{i},status]=elastix(img_array{i},img_array{1},outputDir,pre_align_path,'threads',[]);
end

% Remove temporary directory
rmdir(outputDir,'s')

% Check status and raise error if registration failed
if status ~= 0
    error("Errors occured during registration")
end

end


function [mov_points,atlas_points] = load_points_from_bdv(output_directory,atlas_type,mov_img)
% Function to load points from FIJI's Big Data Viewer

% Update output directory to save points files
output_directory = fullfile(output_directory, 'points_selection');
if ~isfolder(output_directory)
    mkdir(output_directory)
end

files = dir(fullfile(output_directory, '*.csv'));

if isempty(files)
    fprintf("\t Generating default 50 point landmarks file for %s atlas and placing into points_selection directory. "+...
        "Use this as a basis for point selection using Fiji's Big Warp package\n",atlas_type);
    generate_default_points(output_directory,atlas_type)
    % Save also moving image
    save_name = fullfile(output_directory,'landmarks_img_target.tif');
    saveastiff(uint16(mov_img),char(save_name));        
    pause(5)
    error("Exiting")
elseif length(files) >1
    error("Multiple csv files detected in the points_selection directory")
end

pts_path = fullfile(files(1).folder,files(1).name);
pts = readmatrix(pts_path);
pts = pts(:,3:end);

%moving x,y,z then atlas x,y,z
mov_points = pts(:,1:3);
atlas_points = pts(:,4:6);

assert(~all(isnan(atlas_points),'all'), "Not all target points contain coordinates")
assert(~all(isnan(mov_points),'all'), "Not all moving points contain coordinates")

end


function generate_default_points(output_directory,atlas_type)
% Copy pre-defined 50 points to output directory
home_path = fileparts(which('NM_config'));
def_points_path = fullfile(home_path,'data','elastix_parameter_files','atlas_registration','landmarks.csv');
m = readtable(def_points_path);

m2 = m(:,1:2);
if isequal(atlas_type,'nissl')
    m2{:,3:5} = m{:,6:8}; 
    m2 = horzcat(m2,array2table(repmat("Infinity",size(m,1),3),'VariableNames',{'a','b','c'}));
    
    % Move atlas file over to output directory
    atlas_file = fullfile(home_path,'data','atlas','ara_nissl_25.nii');
    img = niftiread(atlas_file);
    
    save_name = fullfile(output_directory,'landmarks_nissl_moving.tif');
    options.message = false;
    options.overwrite = true;
    saveastiff(img,char(save_name),options);
    save_name = 'landmarks_50_nissl.csv';
    
else
    m2{:,3:5} = m{:,3:5};
    m2 = horzcat(m2,array2table(repmat("Infinity",size(m,1),3),'VariableNames',{'a','b','c'}));
    
    % Move atlas file over to output directory
    atlas_file = fullfile(home_path,'data','atlas','average_template_25.nii');
    img = niftiread(atlas_file);
    
    save_name = fullfile(output_directory,'landmarks_average_moving.tif');
    options.message = false;
    options.overwrite = true;
    saveastiff(img,char(save_name),options);
    save_name = 'landmarks_50_average.csv';
end

save_path = fullfile(output_directory,save_name);
writetable(m2,save_path,'WriteVariableNames',0)

end