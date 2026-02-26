function [reg_params_inv,reg_img] = get_inverse_registration_transform(reg_params,target_img_size)
% This function calculate the inverse transform using elastix's
% DisplacementMagnitudePenalty metric. Everything in the inverse transform
% parameter text file should match the original transform except for this
% metric

home_path = fileparts(which('NM_config'));

% Create blank moving image
mov_img = zeros(target_img_size);

% Create temporary directory for saving images
i = 33;
outputDir = fullfile(home_path,'data','tmp',sprintf('tmp%d',i));
if ~exist(outputDir,'dir')
    mkdir(outputDir)
end

% Location of inverse transform file. This should match 
parameter_path{1} = fullfile(home_path,'data','elastix_parameter_files',...
                'atlas_registration','inverse','ElastixParameterBSplineInverse.txt');

% To calculate inverse, only the last transform (B-spline) is needed. But
% any initial rigid transforms need to be saved as text files and specified
% in the parameters structure
if length(reg_params.TransformParameters) > 1
    for i = 1:length(reg_params.TransformParameters)-1
        reg_params.TransformParameters{i}.InitialTransformParametersFileName =...
            'NoInitialTransform';
        fname = fullfile(outputDir,sprintf('init_tform%d.txt',i));
        elastix_paramStruct2txt(fname,reg_params.TransformParameters{i})
        reg_params.TransformParameters{i+1}.InitialTransformParametersFileName = fname;
    end
end
fname = fullfile(outputDir,sprintf('tform_%s.txt','final'));
elastix_paramStruct2txt(fname,reg_params.TransformParameters{end})

% Run the registration using the image of interest as the fixed and moving
% image
[reg_params_inv,reg_img]=elastix(mov_img,mov_img,outputDir,parameter_path,...
    't0',fname,'threads',[]);

% Delete temporary directory
rmdir(outputDir,'s')

% Remove initial transform specification and reset size, spacing to match
% the moving image
reg_params_inv.TransformParameters{1}.InitialTransformParametersFileName =...
    'NoInitialTransform';
%img_size = size(mov_img);
%img_size = img_size([2 1 3]);
%reg_params_inv.TransformParameters{1}.Spacing = img_size ./...
%    reg_params.TransformParameters{end}.Size;

end