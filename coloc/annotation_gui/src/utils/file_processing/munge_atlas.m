function munge_atlas(atlas_file, annotation_file, resolution, orientation, hemisphere, out_resolution)
%--------------------------------------------------------------------------
% Munge custom atlas files and associated annotations and save to /data for
% continued use. Atlas and annotations are typically .nii files and should
% already be perfectly registered to one another. 
%--------------------------------------------------------------------------
% Usage:
% munge_atlas(atlas_file, annotation_file, resolution, orientation,
% hemisphere, out_resolution)
%
%--------------------------------------------------------------------------
% Inputs:
% atlas_file: (string) Full path to atlas file.
%
% annotation_file: (string) Full path to associated annotations.
%
% resolution: (1x3 numeric) Atlas y,x,z resolution specified as micron per
% voxel. (i.e. [25, 25, 25] for 25 um^3 isotropic)
%
% orientation: (1x3 char) Atlas orientation (a/p,s/i,l/r). (i.e. the
% default 'ara_nissl_25.nii' is specified as 'ail'. Note: orientation may
% be flipped based on unit type. Open any .nii file in MATLAB to ensure
% that you're looking at the orientation that would be read when munging.
%
% hemisphere: ("left","right","both","none") Which brain hemisphere.
%
% out_resolution: (int) Isotropic resolution of atlas output. (default: 25)
%
%--------------------------------------------------------------------------

if nargin<6
    out_resolution = 25;
end

% Read images
img = read_img(atlas_file);
annotations = read_img(annotation_file);

assert(all(size(img) == size(annotations)), "Atlas and annotations must be the same size and orientation");

% Standardize
img = standardize_nii(img, resolution, orientation, hemisphere, false,...
    out_resolution, 'ail', hemisphere, 'uint16');

annotations = standardize_nii(annotations, resolution, orientation, hemisphere, true,...
    out_resolution, 'ail', hemisphere, 'uint16');

[~, fname] = fileparts(annotation_file);
annotationData.name = string(fname);
annotationData.annotationVolume = annotations;
annotationData.annotationIndexes = unique(annotationData.annotationVolume);
annotationData.resolution = out_resolution;
annotationData.hemisphere = hemisphere;

if ~isequal(hemisphere, 'none')
    annotationData.orientation = 'ail';
else
    annotationData.orientation = 'none';
end

% Save files
home_path = fileparts(which('NM_config'));
[~,filename] = fileparts(atlas_file);
filepath = fullfile(home_path,'data','atlas',strcat(filename,'.nii'));
niftiwrite(img,filepath)

filepath = fullfile(home_path,'data','annotation_data',strcat(fname,'.mat'));
save(filepath,'-struct','annotationData')

end