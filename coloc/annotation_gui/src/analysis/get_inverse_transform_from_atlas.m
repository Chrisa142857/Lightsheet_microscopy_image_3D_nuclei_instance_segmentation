function [reg_params_inv,reg_img] = get_inverse_transform_from_atlas(config, mov_img, reg_params, direction,inverse_params)
%--------------------------------------------------------------------------
% Calculates the inverse of image_to_atlas transforms to go atlas_to_image or
% vice versa using elastix's DisplacementMagnitudePenalty metric. This
% metric doesn't give an exact inverse but an approximation, which is still
% usually pretty good. To make the approximation more precise, you can
% decrease the grid spacing in the parameter file.
%
% Important: size parameter in the inverse parameters needs to be adjusted
% to match the size of the non-input image.
%--------------------------------------------------------------------------
% Usage: 
% [reg_params_inv,reg_img] = get_inverse_transform_from_atlas(mov_img, 
% config, reg_params, direction);
%--------------------------------------------------------------------------
% Inputs:
% config: (structure) Configuration structure from NM_analyze.
%
% mov_img: (3d matrix) Image to calculate inverse from. For direction
% 'atlas_to_image', this should be the atlas image. For direction,
% 'image_to_atlas', this should be the moving image.
% 
% reg_params: (structure) Elastix registration parameters to invert. Will
% attemp to load from variables if left empty.
%
% direction: ('atlas_to_image','image_to_atlas') Direction of registration.
%
% inverse_params: (string) Folder location containing elastix parameters
% in /data/elastix_parameter_files/atlas_registration to calculate inverse. 
% (default: "inverse")
%
%--------------------------------------------------------------------------
% Outputs:
% reg_params: (structure) Update elastix registration parameters with
% inverse parameters.
%
% reg_img: (3d matrix) Input image registered back to itself using the
% calculated inverse parameters.
%--------------------------------------------------------------------------

if nargin<5
    inverse_params = "inverse";
end

% Note: this is the default parameter file for calculating the inverse
home_path = fileparts(which('NM_config'));
parameter_path{1} = fullfile(home_path,'elastix_parameter_files',...
                'atlas_registration',inverse_params,...
                'ElastixParameterBSplineInverse.txt');

% Load registration parameters if not provided
if isempty(reg_params)
    reg_params = load(fullfile(config.output_directory,'variables','reg_params.mat'),'reg_params');
end
            
% Subset which parameters to inverse
if isequal(direction, "image_to_atlas")
    reg_params_sub = reg_params.atlas_to_img;
elseif isequal(direction, "atlas_to_image")
    reg_params_sub = reg_params.img_to_atlas;
else
    error("Incorrect direction specified")
end

% Create temporary directory for saving images
outputDir = fullfile(config.output_directory,sprintf('tmp_reg_inv_%d',randi(1E4)));
if ~exist(outputDir,'dir')
    mkdir(outputDir)
end

% To calculate inverse, only the last transform (B-spline) is needed. But
% any initial rigid transforms need to be saved as text files and specified
% in the parameters structure
if length(reg_params_sub.TransformParameters) > 1
    for i = 1:length(reg_params_sub.TransformParameters)-1
        reg_params_sub.TransformParameters{i}.InitialTransformParametersFileName =...
            'NoInitialTransform';
        fname = fullfile(outputDir,sprintf('init_tform%d.txt',i));
        elastix_paramStruct2txt(fname,reg_params_sub.TransformParameters{i})
        reg_params_sub.TransformParameters{i+1}.InitialTransformParametersFileName = fname;
    end
end

fname1 = fullfile(outputDir,sprintf('tform_%s.txt','final'));
elastix_paramStruct2txt(fname1,reg_params_sub.TransformParameters{end})

[reg_params_inv,reg_img]=elastix(mov_img,mov_img,outputDir,parameter_path,...
    't0',fname1,'threads',[]);

% Remove initial transform specification
reg_params_inv.TransformParameters{1}.InitialTransformParametersFileName =...
    'NoInitialTransform';

rmdir(outputDir,'s')

end