%% 3D Hessian Method
function centroids = hessian_centroid(img)
%Scales and thresholds
factor = 0.175;
ave_diameter = 8;
scales = linspace(0.8*ave_diameter*factor, 1.25*ave_diameter*factor, 2);
threshold = -0.05;
thresholds = threshold*(1:3);

%resolution = [0.75, 0.75, 2.5]; %resolution = [0.9375, 0.9375, 3.125];
%resolution_test = [0.75, 0.75, 2.5]; %resolution_test = [1 1 2.5];

resolution = [0.75, 0.75, 2.5];
resolution_test = [1 1 2.5];

rescale_resolution_factor = (resolution_test./min(resolution_test)).^-1;

I_scaling = size(img).*(resolution./resolution_test);
I = imresize3(img,I_scaling);
I = im2uint8(I);
I = single(I);

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
centroids = round(centroids.*shrink_factor)';

centroids2 = [centroids(:,2),centroids(:,1),centroids(:,3)];
centroids = centroids2;

end