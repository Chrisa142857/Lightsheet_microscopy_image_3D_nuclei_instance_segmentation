% Create 3D visualization of the Top1 deformation
%    '/Volumes/GoogleDrive/My Drive/Projects/iDISCO/Registration/Jacobians'

path_to_jacobians = 'jacobians_old';
path_to_annotation = 'annotation_25-left_new.nii';
cortex_annotations = 'cortex.csv';

%% Read spatial jacobians and calculate volume difference
A = niftiread(path_to_annotation);
[nrows, ncols, nslices] = size(A);
annot = readtable(cortex_annotations);

files = dir(path_to_jacobians);
w_files = files(arrayfun(@(s) contains(s.name,'WT') ,files));
WT = zeros(nrows,ncols,nslices);

for i = 1:length(w_files)
   img = niftiread(fullfile(w_files(i).folder,w_files(i).name));
   img = imrotate(img,90);
   img = flip(img,1);
   WT = WT + img;
end
WT = WT/length(w_files);

t_files = files(arrayfun(@(s) contains(s.name,'TOP') ,files));
TOP = zeros(nrows,ncols,nslices);

for i = 1:length(t_files)
   img = niftiread(fullfile(t_files(i).folder,t_files(i).name));
   img = imrotate(img,90);
   img = flip(img,1);
   TOP = TOP + img;
end
TOP = TOP/length(t_files);

% Create a difference image
I = (TOP - WT)./WT;

%% Downsample images and generate isosurfaces
padsize = 0;
I2 = imresize3(I,0.5);
A2 = imresize3(A,0.5,'Method','nearest')-1;
%I2 = zeros(size(A2));

% Set vertices outside of cortex to 0
C = single(~ismember(A2,annot.index-1));
C(A2==0) = -1;
C = padarray(C,padsize/2,-1,'both');

I2(~ismember(A2,annot.index-1)) = -1;
I2(A2==0) = -1;
I2 = padarray(I2,padsize/2,-1,'both');

% Erode edges to see deformations along the surface
se = strel('sphere',3);
I2e = imdilate(I2,se);
C2 = imerode(C,se);

% Starting creating the visualization
s = isosurface(I2,-1);
s2 = smoothpatch(s,1,5,1);
idx = sub2ind(size(I2), s.vertices(:,2), s.vertices(:,1), s.vertices(:,3));
colors = I2e(round(idx));

s3 = isosurface(C,0);
s3 = smoothpatch(s3,1,5,1);

%% Plotting
figure
hold on
colormap(flipud(gray_to_blue(100)))
p = patch(s2,...
    'FaceColor','interp',...
    'CData', colors',...
    'EdgeColor','none',...
    'FaceLighting','flat');
isonormals(I2e,p)

p2 = patch(s3,...
    'FaceColor',[0.75 0.75 0.75],...
    'EdgeColor','none',...
    'FaceLighting','flat');
isonormals(C,p2)
material 'dull'
axis vis3d off
daspect([1 1 1])
p.AmbientStrength = 0.7;
p.DiffuseStrength = 0.5;
p2.AmbientStrength = 0.7;
p2.DiffuseStrength = 0.5;

% For general cortex outline
%p.FaceColor = [0.43 1 0.44];
%p.FaceAlpha = 0.6;


% Camera views
% Standing up
%view([0,180])
%camroll(270)

% Sagittal
view([-180,-90])
%rotate([p, p2],[1 1 0],50)

%view([0, 90])
%rotate([p, p2],[0 1 0],40)
%rotate([p, p2],[1 0 0],20)

%camlight('headlight','local')
h = camlight('headlight','local');
h.Style = 'local';
caxis([-1 -0.5])
c = colorbar('horz');
%c.Label.String = 'Percent Change in Volume';
%c.Label.FontSize = 14;
c.Ticks = [-1, -.9, -.8, -.7, -.6, -.5];
c.TickLabels = {'-100','-90','-80','-70','-60','-50'};
c.FontSize = 12;
c.FontName = 'Arial';
hold off

%
A2 = A-1;
%%

% Set vertices outside of cortex to 0
C = single(~ismember(A2,annot.index-1));
C(A2==0) = -1;
C = uint8(C+1);

save_name = 'non-cortex.tif';
imwrite(C(:,:,1),save_name)
for i = 2:size(C,3)
   imwrite(C(:,:,i),save_name,'WriteMode','append'); 
end

