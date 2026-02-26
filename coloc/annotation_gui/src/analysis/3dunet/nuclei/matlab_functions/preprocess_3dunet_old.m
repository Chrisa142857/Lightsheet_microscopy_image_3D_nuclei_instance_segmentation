% Preprocess images for running 3dunet
% Normalize images
% Identify nuclei borders, fills, and centroids
patch_size = [224 224 64];
patch_resolution = [1.21, 1.21, 4];
capture_resolution = [0.75, 0.75, 2.5];

%img_location = './Updated Training Samples/f121_images_224_64/';
img_location = './Updated Training Samples/c121_images_224_64';
cen_location = './Updated Training Samples/c121_centroids_224_64';

%img_location = '/Volumes/GoogleDrive/My Drive/Projects/iDISCO/3DUnet/Updated Training Samples/CUBIC Pretrace/121-Top-Expanded/Images';
%cen_location = '/Volumes/GoogleDrive/My Drive/Projects/iDISCO/3DUnet/Updated Training Samples/CUBIC Pretrace/121-Top-Expanded/Centroids';

save_directory = 'preprocessed';

if ~exist(save_directory,'dir')
    mkdir(save_directory)
end

files = dir(img_location);
file_idx = arrayfun(@(s) contains(s.name,'].nii'),files);
img_files = files(file_idx);

%% Image pre-processing
% Resize and scale intensities
files = dir(img_location);
file_idx = arrayfun(@(s) contains(s.name,'].nii'),files);
img_files = files(file_idx);

for n = 13%:length(img_files)
img_path = fullfile(img_location,img_files(n).name);
img = niftiread(img_path);
img = uint16(img);
img_size = size(img);

se = strel('disk',40);
for z = 1:size(img,3)
   %img(:,:,z) = imtophat(img(:,:,z),se);
   %img(:,:,z) = imgaussfilt(img(:,:,z),1);
end

% Adjust intensity
int_low = stretchlim(img(:))';
int_high = double(max(img(:)))/65535;
img_adj = imadjust3(img,[int_low(1), int_high(1)]);
imshow(img_adj(:,:,10))

% Resize to patch
img_adj = imresize3(img_adj,patch_size,'Method','cubic');

% Write Images
filename_comp = strsplit(img_files(n).name,{'-' '.'});
save_img_directory = fullfile(pwd,save_directory,[filename_comp{1:2}]);

if ~exist(save_img_directory,'dir')
    mkdir(save_img_directory)
end

niftiwrite(img_adj,fullfile(save_img_directory,'t1'),'Compressed',true)
end

%% Generate labels
% 3x3x1 boxed shaped centroid
cen_location = './Updated Training Samples/c121_cen_final_224_64';
rewrite_centroids = 'False';
n_pix = 4;

files = dir(cen_location);
file_idx = arrayfun(@(s) contains(s.name,'].nii'),files);
cen_files = files(file_idx);

se1 = [1 1 1; 1 1 1; 1 1 1];
se2 = [0 1 1 1 0; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 0 1 1 1 0];
n_cells = 0;
for n = 1:length(cen_files)
    save_folder = dir(fullfile(pwd,save_directory));
    img_path = fullfile(save_folder(n).folder,...
        save_folder(n).name,'t1.nii.gz');
    % Load preprocessed image if it exists
    if exist(img_path,'file')
        I = niftiread(img_path);
    else
        img_path = fullfile(img_location,img_files(n).name);
        I = niftiread(img_path);
    end
    % Load labels
    label_path = fullfile(cen_location,cen_files(n).name);
    try
        L = niftiread(label_path);
    catch
        L = niftiread(sprintf('%s.gz',label_path));
    end
    if isequal(rewrite_centroids,'True')
        res = capture_resolution./patch_resolution;
        L_new = zeros(patch_size);
        cc = bwconncomp(L,6);
        label_matrix = labelmatrix(cc);
        
        rp = regionprops(label_matrix);
        
        for j = 1:length(rp)
            pos = round(rp(j).Centroid.*res);
            L_new(pos(2),pos(1),pos(3)) = j;
        end
        
        L_new = expand_centroids3(label_matrix,I,rp);
        %L_new = expand_centroids2(L_new, I, n_pix);
        %L_new2 = erode_centroids(L_new,rp, se1, se2);
        L = uint8(L_new);
    end
    
    L = uint8(L>0);
    
    labels = bwconncomp(L,6);
    n_cells = n_cells + labels.NumObjects;
    
    % Write Images
    filename_comp = strsplit(cen_files(n).name,{'-' '.'});
    save_img_directory = fullfile(pwd,save_directory,[filename_comp{1:3}]);
    
    niftiwrite(L,fullfile(save_img_directory,'truth'),'Compressed',true)
end

%% Split Images Along Z
% Split preprocessed images into smaller chunks
split_size = [224 224 32];
split_idx = [1,4,5,9];

files = dir(save_directory);
files = files(arrayfun(@(s) ~contains(s.name,'split'),files));

for n = split_idx
    img_path = fullfile(save_directory,files(n).name,'t1.nii.gz');
    img = niftiread(img_path);
    
    cen_path = fullfile(save_directory,files(n).name,'truth.nii.gz');
    cen = niftiread(cen_path);

    z_pos = 1:split_size(3):size(img,3)+1;
    
    for j = 1:length(z_pos)-1
        img_chunk = img(:,:,z_pos(j):z_pos(j+1)-1);
        cen_chunk = cen(:,:,z_pos(j):z_pos(j+1)-1);
        
        filename_comp = strsplit(img_files(n).name,{'-' '.'});
        save_img_directory = fullfile(pwd,save_directory,sprintf('%s_split%d',...
            [filename_comp{1:3}],j));

        if ~exist(save_img_directory,'dir')
            mkdir(save_img_directory)
        end
        
        niftiwrite(img_chunk,fullfile(save_img_directory,'t1'),'Compressed',true)    
        niftiwrite(cen_chunk,fullfile(save_img_directory,'truth'),'Compressed',true)    
    end
end

%% Redraw centroids
img_location = 'prediction';
save_directory = 'redrawn';
threshold = 0.01;

cen_location = './Updated Training Samples/c121_cen_expanded_224_64';
cenIdx = 9;

pre_files = dir(img_location);

% Create pre-traced image
for n = 1:length(pre_files)
    img_path = fullfile(img_location,pre_files(n).name,'prediction.nii');
    img_chunk = niftiread(img_path);
    
    if n == 1
        img = img_chunk;
    else
        img = cat(3,img,img_chunk);
    end
end

cen_files = dir(cen_location);
cen_path = fullfile(cen_files(cenIdx).folder,...
    cen_files(cenIdx).name);

cen_raw = niftiread(cen_path);
cen_new = zeros(size(cen_raw));

for i = 1:size(cen_raw,3)
    cen_slice = cen_raw(:,:,i);
    pre_slice = img(:,:,i);
    new_slice = cen_slice;
    
    pre_slice = imbinarize(pre_slice,threshold);
    
    pre_slice = bwareaopen(pre_slice,3);
    
    cen_objects = bwconncomp(cen_slice,4);
    pre_objects = bwconncomp(pre_slice,4);
    pre_label = labelmatrix(pre_objects);

    for j = 1:cen_objects.NumObjects
        cen_pix = cen_objects.PixelIdxList{j};
        overlap_label = pre_label(cen_pix);
        
        labels = unique(overlap_label);
        labels = labels(labels>0);
        
        for k = 1:length(labels)
            try
                new_slice(pre_label == labels(k)) = 1;
            catch
                a = 1;
            end
                
        end
    end
    
    cen_new(:,:,i) = new_slice;
end

niftiwrite(uint8(cen_new),fullfile(save_directory,...
    sprintf('%s_exapnded',cen_files(cenIdx).name(1:end-7))),...
    'Compressed',true)    


%%
    %labels = unique(L);
    %centroid_img = zeros(size(L));
    %if length(labels) > 2 % Each cell has unique label
    %    L = imresize3(L,patch_size,'Method','nearest');
    %    centroid_img = zeros(size(L));
    %    for i = 2:length(labels)
    %        img0 = zeros(size(L));
    %        idx = find(L == labels(i));
    %        img0(idx) = 1;
    %    
    %        if sum(img0(:) ~= 0)
    %            index = find(img0 == 1);
    %            [x, y, z] = ind2sub(size(img0),index);
    %            centroid_img(round(mean(x)),round(mean(y)),round(mean(z))) = 1;
    %        end
    %    end
    %else %Each cell just has centroid
    %    bw = bwconncomp(L);
    %    labeled = labelmatrix(bw);
    %    rp = regionprops(labeled);
    %    
    %    res = patch_size./img_size;
    %    cen_res = arrayfun(@(s) s.Centroid.*res,rp,'UniformOutput',false);        
    %    
    %    centroid_img = imresize3(centroid_img,patch_size,'Method','nearest');
    %    for i = 1:length(cen_res)
    %        out = round(cen_res{i});
    %        centroid_img(out(2),out(1),out(3)) = 1;
    %    end
    %end        
    
    %centroid_dilated = imdilate(centroid_img,se);
    %cen_labels = bwconncomp(centroid_dilated,6);
    %disp(cen_labels.NumObjects)
    
    %% Image Augmentation
% Augment by resizing to captured resolution then back to patch resolution

files = dir(save_directory);
scale_factor = patch_resolution./capture_resolution;
resize_patch_size = round(patch_size.*scale_factor);

for n = 1:length(files)
    img_path = fullfile(save_directory,files(n).name,'t1.nii.gz');
    img = niftiread(img_path);
    
    down_img = imresize3(img, resize_patch_size, 'Method', 'cubic');
    up_img = imresize3(down_img, patch_size, 'Method', 'cubic');
    
    up_img = imgaussfilt3(up_img);
    
    % Write Images
    filename_comp = strsplit(img_files(n).name,{'-' '.'});
    save_img_directory = fullfile(pwd,save_directory,sprintf('%s_augmented',[filename_comp{1:3}]));

    if ~exist(save_img_directory,'dir')
        mkdir(save_img_directory)
    end

    niftiwrite(up_img,fullfile(save_img_directory,'t1'),'Compressed',true)    
end

