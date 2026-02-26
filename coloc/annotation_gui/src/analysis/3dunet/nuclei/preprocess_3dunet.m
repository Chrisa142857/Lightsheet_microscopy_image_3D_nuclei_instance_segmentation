function preprocess_3dunet(part,file_delimiter,n)
%--------------------------------------------------------------------------
% Preprocess images for running 3dunet. Normalize images. Identify nuclei 
% borders, fills, and centroids. Perform morphological operations on
% centroid patches. Save images in preprocessed directory.
%
% Images (.nii 16-bit) should be placed in /data/raw/(partition) with at 
% least 1 folder containing raw images (i) and 1 folder containing centroid 
% patches (c) with unique indexes. Image and centroids filenames should 
% start with an index to matchup images with centroids. Typically the 
% following nomeclature is used: 
%   i.e. /data/raw/075/c0201_xxxx.nii: 
%       Image patch #2 from image set #1 in '075' partition.
%       Corresponding centroid patch: c0201_xxxx.nii
%
% Note patches are read in alphabetical and are not shuffled during 
% training unless specified here. If there are few training images sampled 
% from many regions, it becomes more important to get adequate
% representation of all image features in both training and testing sets, 
% which may be inconsistent if patches are shuffled.
%
%--------------------------------------------------------------------------
% Usage:
% preprocess_3dunet(part,file_delimiter,n)
%
%--------------------------------------------------------------------------
% Inputs:
% part: (string) Read only certain partition. (default: [], read all 
% partitions)
%
% file_delimiter: (string) Keep only files with this delimiter in the
% filename. (i.e. "final" keeps only images with "final" in name).
% (default: [], use all files)
%
% n: (int) Limit number of images read (for testing purposes). (default:
% [], use all images)
%
%--------------------------------------------------------------------------

final_size = [112 112 32];      % Final patch size. Images will be cropped or padded
shuffle_images = false;         % Shuffle images

rescale_centroids = true;      % Resample patches to specific resolution and rewrite centroids   
capture_res = [1.21,1.21,4];  % Resolution at which images were acquired
patch_res = [1.8,1.8,4];    % Resolution to resample to (Resampling of z dimension is not recommended)

erode_blobs = true;         % Erode pixels around the edges
pix_threshold = 0.4;        % Erode fraction pixels from farthest from center
normalize_images = true;    % Perform min/ax normalization
remove_background = false;  % Perform background subtraction
write_edges = false;        % Add edge labels as index=2
size_threshold = 0;

% Create preprocessed directory
home_path = fileparts(which('preprocess_3dunet'));
save_directory = fullfile(home_path,'data','preprocessed');
if ~isfolder(save_directory)
    mkdir(save_directory)
elseif ~isempty(dir(save_directory))
    rmdir(save_directory,'s')
    mkdir(save_directory)
end

% Subset parition
folders = dir(fullfile(home_path,'data','raw'));
folders = folders(arrayfun(@(s) ~ismember(s.name,{'.','..'}),folders)); 
if nargin < 1
    folders = folders([folders.isdir]);
else
    folders = folders(arrayfun(@(s) isequal(s.name,part),folders));
end

% Load images and match centroid files
img_files = cell(length(folders),1);
for i = 1:length(folders)
    files = dir(fullfile(folders(i).folder, folders(i).name));
    files = files(arrayfun(@(s) contains(s.name,'.nii'),files));
    
    % Trim files if delimiter is provided
    if nargin>1
        files = files(arrayfun(@(s) contains(s.name,file_delimiter),files));
    end
    
    % Match image files with centroid files in this partition
    ifiles = files(arrayfun(@(s) startsWith(s.name,'i'),files));
    filenames = {ifiles.name};
    ifiles = arrayfun(@(s) fullfile(s.folder,s.name),ifiles,'UniformOutput',false);
    cfiles = cell(length(ifiles),1);
    for j = 1:length(ifiles)
        idx = strsplit(filenames{j},'_');
        c_idx = ['c',idx{1}(2:end)];
        idx = startsWith({files.name},c_idx);
        cfiles{j} = fullfile(files(idx).folder,files(idx).name);
    end
    img_files{i} = [ifiles,cfiles];
end
img_files = cat(1,img_files{:});
n_images = size(img_files,1);

% Shuffle images
if shuffle_images
    img_files = img_files(randperm(n_images),:);
end

% Subset images
if nargin>2
    idx = randperm(n_images);
    img_files = img_files(idx(1:n),:);
end

% Load images, normalize, and save
n_cells = 0;
pixels = zeros(1,n_images);
for i = 1:n_images
    % Read image
    [~,img_name] = fileparts(img_files{i,1});
    fprintf('%s\n',img_name)
    I = niftiread(img_files{i,1});
    I = uint16(I);
    
    % Read centroids
    [~,cen_name] = fileparts(img_files{i,2});
    fprintf('%s\n',cen_name)
    L = niftiread(img_files{i,2});
    L = uint16(L);
    
    % Normalize image
    if normalize_images
        I = normalize_image(I,remove_background);
    end

    % Take 2D slice if 3D indexed centroid volume provided
    if length(unique(L(:)))>2
        rp = regionprops3(L,I,'WeightedCentroid','VoxelList');
        rp =rp(any(~isnan(rp.WeightedCentroid),2),:);
        L = zeros(size(L),'uint16');
        for j = 1:height(rp)
            z_slice = round(rp(j,:).WeightedCentroid);
            vox = rp(j,:).VoxelList{1};
            vox = vox(vox(:,3) == z_slice(3),:);
            vox = sub2ind(size(L),vox(:,2),vox(:,1),vox(:,3));
            L(vox) = 1;
        end
        % Increase erosion for fully traced nuclei
        pix_threshold = pix_threshold - (1-pix_threshold)*0.25;
    end
    
    % Rescale centroids
    if rescale_centroids
        res_adj = capture_res./patch_res;
        I = imresize3(I,round(res_adj.*size(I)),'Method','cubic');
        if res_adj(3) == 1
            L = imresize3(L,round(res_adj.*size(L)),'Method','nearest');
        else
            error("Unfortunately z dimension cannot be resampled")
        end
        %L = resample_centroids(L,I,capture_res,patch_res);
    end
    L = uint8(L>0);
    
    % Check orientatioin of centroids to image
    [L,~] = check_orientation(L,I);
    
    % Erode centroid blobs
    if erode_blobs
        L = erode_blob(L,I,size_threshold,pix_threshold);
    end
    
    % Save edge pixel values as 2
    if write_edges
       L = write_edge(L); 
    end
    
    % Count cells
    labels = bwconncomp(L,6);
    n_cells = n_cells + labels.NumObjects;
    
    % Count percent of positive pixels
    pixels(i) = sum(L(:)>0)/numel(L(:));
    
    % Pad or crop images
    pad_val = median(I,'all');
    [I,L] = pad_or_crop_dim(I,L,final_size,3,pad_val);
    [I,L] = pad_or_crop_dim(I,L,final_size,2,pad_val);
    [I,L] = pad_or_crop_dim(I,L,final_size,1,pad_val);
    
        
    % Split to final chunk size and write images
    patch_size = size(I);
    n_chunks = patch_size./final_size;
    x_chunks = floor(linspace(0,patch_size(1),n_chunks(1)+1));
    y_chunks = floor(linspace(0,patch_size(2),n_chunks(2)+1));
    z_chunks = floor(linspace(0,patch_size(3),n_chunks(3)+1));

    a = 1;
    for x = 1:n_chunks(1)
        for y = 1:n_chunks(2)
            for z = 1:n_chunks(3)
                xi = x_chunks(x)+1:x_chunks(x+1);
                yi = y_chunks(y)+1:y_chunks(y+1);
                zi = z_chunks(z)+1:z_chunks(z+1);

                L_chunk = L(xi,yi,zi);
                I_chunk = I(xi,yi,zi);

                % Create new folder in directory
                dir_name = fullfile(save_directory,...
                    sprintf('s%d%s',a,cen_name));
                mkdir(dir_name)

                niftiwrite(L_chunk,fullfile(dir_name,'truth'),'Compressed',true)
                niftiwrite(I_chunk,fullfile(dir_name,'t1'),'Compressed',true)
                a = a+1;
            end
        end
    end

end


fprintf('Preprocessed %d cells in %d images \n',n_cells,i)
fprintf('Percent of pixels: %f \n',mean(pixels)*100)

end

function I = read_nii(path)
    try
        I = niftiread(path);
    catch
        I = niftiread(sprintf('%s.gz',path));
    end
end

function img_adj = normalize_image(img, remove_background)
% Remove background
if remove_background
    se = strel('disk',20);
    for z = 1:size(img,3)
        img(:,:,z) = imtophat(img(:,:,z),se);
    end
end

% Adjust intensity
int_low = double(min(img(:)))/65535;
int_high = double(max(img(:)))/65535;
img_adj = imadjustn(img,[int_low, int_high(1)]);
        
end

function L_new = write_edge(L)
nhood = [0 1 0;1 1 1;0 1 0];
L_bound = zeros(size(L));
for z = 1:size(L,3)
    L_slice = imdilate(L_bound(:,:,z),nhood);
    boundary = bwboundaries(L(:,:,z));
    pos = cell2mat(boundary);
    idx = sub2ind(size(L_slice),pos(:,1),pos(:,2));
    L_slice(idx) = 1;
    L_bound(:,:,z) = L_slice;
end

L_new = L + uint8(L_bound);
end

function [L,I] = check_orientation(L,I)

z = round(linspace(1,size(L,3),5));

a = 1;
score = zeros(1,6);
for i = 1:2
   for j = 1:3
       cc = zeros(1,5);
       for k = 1:5
           if i == 2
            lmod = flip(L(:,:,z(k)),1);
           else
            lmod = L(:,:,z(k));
           end
           
           if j == 2
               lmod = imrotate(lmod,90);
           elseif j == 3
               lmod = imrotate(lmod,-90);
           end

            cc(k) = corr2(lmod,I(:,:,z(k)));
       end
       score(a) = mean(cc);
    a = a+1;
   end    
end

[~,idx] = max(score);

if any(idx == 4:6)
    L = flip(L,1);
end

if any(idx == [2,4])
   L = imrotate(L,90);
elseif any(idx == [3,6])
   L = imrotate(L,-90);
end

end

function L_new = erode_blob(L,I,size_threshold,pix_threshold)

L_new = zeros(size(L));
dim = size(L(:,:,1));

for z = 1:size(L,3)
   L_slice = L(:,:,z);
   I_slice = I(:,:,z);
   cc = bwconncomp(L_slice,4);
   rp = regionprops(cc);
   
   % Erode blobs a certain size
   if size_threshold > 0 
       for k = 1:cc.NumObjects
         L_object = zeros(dim);
         L_object(cc.PixelIdxList{k}) = 1;
        if length(cc.PixelIdxList{k}) > size_threshold
             L_object = L_object - bwperim(L_object); 
        end
        L_new(:,:,z) = L_new(:,:,z) + L_object;
       end
   else
       % Erode some percent of pixels around the centroid
      for k = 1:cc.NumObjects
         L_object = zeros(dim);
         L_object(cc.PixelIdxList{k}) = 1;
         npix = length(cc.PixelIdxList{k});
         pix_to_erode = floor(npix*pix_threshold);
         
         [x1,y1] = ind2sub(dim,cc.PixelIdxList{k});
         cen = rp(k).Centroid;
         D = sqrt(sum(([cen(2),cen(1)] - [x1,y1]).^2, 2));
         [~, idx] = sort(D,'descend');
         
         erode_idx = cc.PixelIdxList{k}(idx(1:pix_to_erode));
         L_object(erode_idx) = 0;

        L_new(:,:,z) = L_new(:,:,z) + L_object;
      end
   end
end

L_new = uint8(L_new);
end

function [I,L] = pad_or_crop_dim(I,L,final_size,dim,pad_val)

s = size(I);
chunks = s(dim)/final_size(dim);
if abs(chunks-round(chunks)) ~= 0
    if abs(chunks-floor(chunks))<0.25
        % Crop
        chunks = floor(chunks);
        pad1 = abs(ceil((s(dim) - final_size(dim)*chunks)/2));
        pad2 = abs(floor((s(dim) - final_size(dim)*chunks)/2));
        if dim==1
            I = I(pad1:end-pad1-1,:,:);
            L = L(pad2:end-pad2-1,:,:);
        elseif dim==2
            I = I(:,pad1:end-pad1-1,:);
            L = L(:,pad2:end-pad2-1,:);
        else
            I = I(:,:,pad1:end-pad1-1);
            L = L(:,:,pad2:end-pad2-1);
        end
    else
        % Pad
        chunks = ceil(chunks);
        pad1 = abs(ceil((s(dim) - final_size(dim)*chunks)/2));
        pad2 = abs(floor((s(dim) - final_size(dim)*chunks)/2));
        if dim==1
            I = cat(1,repmat(pad_val,[pad1,s(2),s(3)]),I);
            I = cat(1,I,repmat(pad_val,[pad2,s(2),s(3)]));
            L = cat(1,zeros([pad1,s(2),s(3)]),L);
            L = cat(1,L,zeros([pad2,s(2),s(3)]));
        elseif dim==2
            I = cat(2,repmat(pad_val,[s(1),pad1,s(3)]),I);
            I = cat(2,I,repmat(pad_val,[s(1),pad2,s(3)]));
            L = cat(2,zeros([s(1),pad1,s(3)]),L);
            L = cat(2,L,zeros([s(1),pad2,s(3)]));
        else
            I = cat(3,repmat(pad_val,[s(1),s(2),pad1]),I);
            I = cat(3,I,repmat(pad_val,[s(1),s(2),pad2]));
            L = cat(3,zeros([s(1),s(2),pad1]),L);
            L = cat(3,L,zeros([s(1),s(2),pad2]));
        end
    end
end

end
