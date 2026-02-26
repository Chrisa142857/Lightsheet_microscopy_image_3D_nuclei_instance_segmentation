function download_atlas_files(atlas,resolution)
%--------------------------------------------------------------------------
% Read atlas .nrrd files and convert to 16-bit .nii to match annotations.
% Atlas files will be places in /data/atlas.
%--------------------------------------------------------------------------
% Usage: 
% download_atlas_files(atlas,resolution)
%
% Inputs:
% atlas - ('average' or 'nissl'). Download MRI average atlas or nissl
% atlas. Default is nissl.
%
% resolution - (default: 25). Atlas isotropic resolution.
%--------------------------------------------------------------------------

if nargin<1
    atlas = 'nissl';
end

if nargin<2
    resolution = 25;
end

if ~isequal(atlas,'average') && ~isequal(atlas,'nissl')
    error("Valid atlas options are 'average' or 'nissl'")
end

if ~ismember(resolution,[10,25,50,100])
    error("Valid resolution options are 10, 25, 50, or 100")
end

average_path = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/";
nissl_path = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/";

% Get web address
if isequal(atlas,'nissl')
    filename = sprintf('ara_nissl_%s.nrrd',num2str(resolution));
    final_path = fullfile(nissl_path,filename);
else
    filename = sprintf('average_template_%s.nrrd',num2str(resolution));
    final_path = fullfile(average_path,filename);
end

% Download file
fprintf('Downloading atlas file...\n')
outfilename = websave(filename,final_path);

% Adjust files
fprintf('Adjusting atlas and saving...\n')
I = uint16(nrrdread(outfilename));

% Adjust intensity
min_int = im2double(min(I(:)));
max_int = im2double(max(I(:)));
I = imadjustn(I,[min_int,max_int]);

% Cut this brain half and save only the left hemisphere
z = round(size(I,3)/2);
I = I(:,:,1:z);
I = imrotate(I,-90);

% Save as .nii
atlas_path = fullfile(pwd,"/data/atlas");
if exist(atlas_path,'dir') ~= 7
    mkdir(atlas_path)
end

[~,name] = fileparts(outfilename);
niftiwrite(I,fullfile(atlas_path,name+".nii"))
delete(outfilename)

end