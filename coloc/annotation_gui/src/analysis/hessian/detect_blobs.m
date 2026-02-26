% Cell detection
function [pixel_idx_list, box_range] = detect_blobs(path_table,I_mask, nuc_diameter_range, resolution, adj_param)


%Scales and thresholds
factor = 0.175;
ave_diameter = 10; %mean(nuc_diameter_range);
scales = linspace(0.8*ave_diameter*factor, 1.25*ave_diameter*factor, 2);
threshold = -0.05;
thresholds = threshold*(1:3);

resolution_test = [1 1 2.5];
rescale_resolution_factor = (resolution_test./min(resolution_test)).^-1;
chunk_size = 100;
pad = 4;

%Load images
tic
num_images = height(path_table);
tempI = imread(path_table.file{1});
[nrows, ncols] = size(tempI);
%I = uint16(zeros(nrows,ncols,num_images));

for i = 1:num_images
   %I(:,:,i) = loadtiff(path_table.file{i});
end
fprintf('Images loaded in: %d seconds\n',toc)

%Some pre-processing
tic
%Imax = prctile(double(I(:)),99)/65535;
Imax = double(0.0739);
I_mask = imresize3(single(I_mask),[size(I_mask,1),size(I_mask,2),num_images],'method','nearest');

clear I

% Chunk sizes
chunks = round(floor(linspace(1,num_images,50)));
chunks(end) = num_images;
pixel_idx_list = cell(1,length(chunks-1));


%Start parallel pool
p = gcp('nocreate');
if isempty(p)
    %parpool(2)
end

box_range = cell(length(chunks)-1);

for i = 1:length(chunks)-1
    fprintf('Analyzing chunk: %d out of %d\n',i,length(chunks-1))
    tic
    chunk_start = chunks(i);
    chunk_end = min(num_images, chunks(i+1)+pad);
    
    %Apply mask 
    I_mask_sub = I_mask(:,:,chunk_start:chunk_end);
    I_mask_sub = imresize3(I_mask_sub,[nrows ncols size(I_mask_sub,3)],'method','nearest');
    I_mask_sub = logical(I_mask_sub);

    I = uint16(zeros(nrows,ncols,length(chunk_start:chunk_end)));
    a = 1;
    for j = chunk_start:chunk_end
        I(:,:,a) = loadtiff(path_table.file{j});
        a = a+1;
    end
    
    %This is to normalize intensity based on whole image
    %high_in_uint16 = 2000;
    %I_sort = sort(I(:));
    %[~,min_idx] = min(abs(double(I_sort)-high_in_uint16));
    %high_in = min_idx/length(I_sort);
    
    %low_in_uint16 = 180;
    %[~,min_idx] = min(abs(double(I_sort)-low_in_uint16));
    %low_in = min_idx/length(I_sort);
    
   %clear 'I_sort'
    %%%
    I(~I_mask_sub) = 0;
    
    proj = max(I,[],3);
    [rows_sub,cols_sub] = find(proj);
    
    top = min(rows_sub);
    bottom = max(rows_sub);
    left = min(cols_sub);
    right = max(cols_sub);
    
    box_range{i} = [top bottom left right];
    
    I = I(top:bottom,left:right,:);
    I = imadjustn(I);
    
    I_scaling = size(I).*(resolution./resolution_test);
    I = imresize3(I,I_scaling);
    
    I_mask_sub = parclear(I_mask_sub);
    I = im2uint8(I);

    I = single(I);

fprintf('Image pre-processing: %d seconds\n',toc)
I_max = max(I(:));
seeds = ones(size(I));

%Detect cells from Hessian eigenvalues
for scale = scales
    tic
    %Sigma for given scale
    sigmas = scale*rescale_resolution_factor;
    
    % Apply 3D filter according to sigma
    I_filt = imGaussianFilter(I,round(sigmas*3),sigmas,'symmetric');
    I_filt = normalize_mins(I_filt) * I_max;

    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = imHessian(I_filt, 0.9);
    I_filt = parclear(I_filt);
    [l1,l2,l3]=eig3volume(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz);
    
    seeds = and(seeds, l1 < thresholds(1));
    seeds = and(seeds, l2 < thresholds(2));
    seeds = and(seeds, l3 < thresholds(3));
    
    fprintf('Image analysis: %d seconds\n',toc)
    %subplot(1,2,1); imagesc(I_filt(:,:,5)); title('first image');
    %subplot(1,2,2); imagesc(seeds(:,:,17)); title('second image');
    %subplot(1,3,1); imagesc(l1(:,:,5))
    %subplot(1,3,2); imagesc(l2(:,:,5))
    %subplot(1,3,3); imagesc(l3(:,:,5))
end


%Closing
tic
seeds = imclose(seeds, true(1, 1, 3));
seeds = imclose(seeds, true(1, 3, 1));
seeds = imclose(seeds, true(3, 1, 1));

%Opening
struc = false(3, 3, 3);
c = (3+1)/2;
struc(:, c, c) = true; struc(c, :, c) = true; struc(c, c, :) = true;
seeds = imopen(seeds, struc);

%Remove small seeds
%seeds = bwareaopen(seeds,7,26);

CC = bwconncomp(seeds,26);

ave_int = 15;

for j = 1:CC.NumObjects
    ind = CC.PixelIdxList{j};
    pixels = I(ind);
   
    if max(pixels) < ave_int*2
        seeds(ind) = 0;
    end
end

rp = regionprops(seeds,'Centroid');
centroids = reshape([rp.Centroid],3,length([rp.Centroid])/3);
shrink_factor = (resolution_test./resolution)';
centroids = round(centroids.*shrink_factor);

centroids(1,:) = centroids(1,:)+left-1;
centroids(2,:) = centroids(2,:)+top-1;
centroids(3,:) = centroids(3,:)+chunk_start-1; 

disp(size(centroids,2))

pixel_idx_list{i} = centroids;
parsave(centroids,chunks(i))
end

L = zeros(size(I));
index = randperm(CC.NumObjects);
for k = 1:CC.NumObjects
   ind = CC.PixelIdxList{k};
   L(ind) = index(k);
end

L2 = label2rgb3d(L,'jet');
L2 = imcomplement(L2);

for k = 1:size(L2,3)
   slice = cat(3,L2(:,:,k,1),  L2(:,:,k,2),  L2(:,:,k,3));
    imwrite(slice,sprintf('slice_%d.tif',k))
end

end

function data  = parclear(data)
    clear('data')
    data =[];
end

function parsave(centroid,index)
    name = sprintf('TOP_centroids_%d.mat',index);
    save(name,'centroid')
end
