function def_field = get_deformation_field(reg_params,resolution)
% Compute the deformation field at voxel position from registration 
% parameters by calculating the determinants of the spatial jacobian in 
% elastix. Direction should typically be image_to_atlas to get image
% deformation relative to atlas. 

if nargin<2
    resolution = 100;
end

% Direction should be image to atlas
def_field = transformix([],reg_params,[1,1,1],[],'jac');

% Resize deformation field
res_adj = 25/resolution;
def_field = imresize3(def_field,res_adj,'Method','linear');

% Negative values are not good.
% Set negative value to 0
%def_field(def_field<0) = 0;

end