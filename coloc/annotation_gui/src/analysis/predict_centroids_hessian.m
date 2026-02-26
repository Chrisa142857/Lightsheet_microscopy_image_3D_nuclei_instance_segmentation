function predict_centroids_hessian(config, path_table,path_save,I_mask)
% Cell (blob) detection calculated by measuring hessian determinant

% Unpack parameters from config
chunk_size = config.chunk_size;
overlap = config.chunk_overlap;
res = config.resolution{1};
min_int = config.min_intensity;

if isequal(config.use_annotation_mask,"true")
    mask_res = config.resample_resolution;    
end

%Scales and thresholds
factor = 0.175;
ave_diameter = config.average_nuc_diameter;
scales = linspace(0.8*ave_diameter*factor, 1.25*ave_diameter*factor, 2);
threshold = -0.05;
thresholds = threshold*(1:3);

%resolution_test = [1 1 2.5];
rescale_resolution_factor = (res./min(res)).^-1;

% Image info
num_images = height(path_table);
tempI = imread(path_table.file{1});
[nrows, ncols] = size(tempI);

% Intensity threshold
if isempty(min_int)
    min_int = config.signalThresh(1);
    min_int = (min_int-config.lowerThresh(1))/(config.upperThresh(1)-config.lowerThresh(1));
elseif min_int>0
    if min_int>1
        min_int = min_int/65535;
    end
    min_int = (min_int-config.lowerThresh(1))/(config.upperThresh(1)-config.lowerThresh(1));
else
    min_int = 0;
end

% Get chunk z ranges
chunk_ranges = get_chunk_ranges(1,num_images,chunk_size(3)-overlap(3),overlap(3),num_images);
n_chunks = size(chunk_ranges,1);

% Get chunk xy grid ranges
% Calculate number of chunks
y_tiles = ceil((nrows+overlap(1))/(chunk_size(1)-overlap(1)));
x_tiles = ceil((ncols+overlap(2))/(chunk_size(2)-overlap(2)));

% Calculate x,y chunk positions with specified overlap
y_start_adj = 1:chunk_size(1)-overlap(1):(chunk_size(1)-overlap(1))*y_tiles;
x_start_adj = 1:chunk_size(2)-overlap(2):(chunk_size(2)-overlap(2))*x_tiles;
y_end_adj = y_start_adj+(chunk_size(1))-1;
x_end_adj = x_start_adj+(chunk_size(2))-1;

% Calculate padding
y_pad1 = floor((y_end_adj(end)-nrows)/2);
y_pad2 = ceil((y_end_adj(end)-nrows)/2);
x_pad1 = floor((x_end_adj(end)-ncols)/2);
x_pad2 = ceil((x_end_adj(end)-ncols)/2);

% Linearize to create 4xn_tiles matrix with tile start and end positions
[y1,x1] = meshgrid(y_start_adj,x_start_adj);
[y2,x2] = meshgrid(y_end_adj,x_end_adj);
yx = [y1(:),y2(:),x1(:),x2(:)];

% Adjust mask 
if ~isempty(I_mask)
    [nrow_m,ncol_m,nslice_m] = size(I_mask);
    I_mask2 = imresize3(logical(I_mask),[nrow_m,ncol_m,num_images],'Method','nearest');
end

%Start parallel pool
try 
    p = gcp('nocreate');
catch
    p = [];
end
if isempty(p)
    %parpool(2)
end

cen_count = 0;
for i = 1:n_chunks
    tic    
    chunk_start = chunk_ranges(i,3);
    chunk_end = chunk_ranges(i,4);
    z_pos = chunk_start:chunk_end;
    nslices = length(z_pos);
    fprintf('Analyzing slices: %d to %d\n',chunk_start,chunk_end)
    
    % Read images
    I = zeros(nrows,ncols,nslices,'uint16');
    for j = 1:nslices
        I(:,:,j) = read_img(path_table,[1,z_pos(j)]);
    end
    
    % Adjust mask
    if ~isempty(I_mask)
        I_mask_sub = logical(I_mask2(:,:,z_pos));
        I_mask_sub = imresize3(I_mask_sub,size(I),'Method','nearest');
    else
        I_mask_sub = ones(size(I),'logical');
    end
    
    % Normalize intensity based on full image
    I = imadjustn(I,[config.lowerThresh(1),config.upperThresh(1)]);
    I = single(I);

    % Pad images
    I = padarray(I,[y_pad1,x_pad1,0],0,'pre');
    I = padarray(I,[y_pad2,x_pad2,0],0,'post');
    I_mask_sub = padarray(I_mask_sub,[y_pad1,x_pad1,0],0,'pre');
    I_mask_sub = padarray(I_mask_sub,[y_pad2,x_pad2,0],0,'post');
    
    fprintf('Image pre-processing: %d seconds\n',toc)
    I_max = 65535;
    
    % Detect cells from Hessian eigenvalues
    fprintf('Counting centroids across %d chunks\n',size(yx,1))
    centroids = cell(1,size(yx,1));
    tic
    for j = 1:size(yx,1)
        % If mask chunk is empty, continue to next chunk
        if ~any(I_mask_sub,'all')
            continue
        else
            I_sub = I(yx(j,1):yx(j,2),yx(j,3):yx(j,4),:);
            I_sub = I_sub.*I_mask_sub(yx(j,1):yx(j,2),yx(j,3):yx(j,4),:);
        end
        
        % Get initial seeds
        seeds = get_seeds(I_sub,I_max,scales,thresholds,rescale_resolution_factor);
        
        % Cleanup seeds through various morphological filters
        seeds = cleanup_seeds(seeds);
        
        % Calculate connected components
        CC = bwconncomp(seeds,26);
        
        % Calculate region properties
        % Note centroids given as [x,y,z]
        rp = regionprops(CC,I_sub,'Centroid','MaxIntensity');
        rp = rp(arrayfun(@(s) s.MaxIntensity>min_int*65535,rp),:);
        
        % Not centroids remain after threshold
        if isempty(rp)
            continue
        end
        
        % Merge centroids
        cen = cat(1,rp.Centroid);
        cen = cen(:,[2,1,3]);
        
        % Trim centroids in overlapping regions
        cen = cen(cen(:,1)>overlap(1)/2 & cen(:,1)<=chunk_size(1) - overlap(1)/2,:);
        cen = cen(cen(:,2)>overlap(2)/2 & cen(:,2)<=chunk_size(2) - overlap(2)/2,:);

        % Save
        cen(:,1) = cen(:,1) + yx(j,1) - 1;
        cen(:,2) = cen(:,2) + yx(j,3) - 1;        
        centroids{j} = round(cen);
    end
    fprintf('Centroid detection: %d seconds\n',toc)
    
    % Merge and trim centroids
    centroids = cat(1,centroids{:});
    centroids(:,1) = centroids(:,1) - y_pad1;
    centroids = centroids(centroids(:,1)>0,:);
    centroids(:,2) = centroids(:,2) - x_pad1;
    centroids = centroids(centroids(:,2)>0,:);
    centroids(:,3) = centroids(:,3) + chunk_start-1;
    
    % Remove any residual centroids too close to each other
    %centroids = kd_clean(centroids, 2);
    
    % Add annotations
    if isequal(config.use_annotation_mask,"true")
        adj = res/mask_res;
        yxz = [round(centroids(:,1)*adj(1)),...
            round(centroids(:,2)*adj(2)),...
            round(centroids(:,3)*adj(3))];
        yxz(yxz == 0) = 1;
        yxz(yxz(:,1)>nrow_m,1) = nrow_m;
        yxz(yxz(:,2)>ncol_m,2) = ncol_m;
        yxz(yxz(:,3)>nslice_m,3) = nslice_m;
        a_idx = sub2ind([nrow_m,ncol_m,nslice_m],yxz(:,1),yxz(:,2),yxz(:,3));
        centroids(:,4) = I_mask(a_idx);
    else
        centroids(:,4) = 1;
    end
    
    cen_count = cen_count+size(centroids,1);
    fprintf('Centroids counted in chunks: %d\n',size(centroids,1))
    fprintf('Total centroids counted: %d\n',cen_count)

    % Write to file
    if ~isfile(path_save)
        writematrix(centroids,path_save)
    else
        writematrix(centroids,path_save,'WriteMode','append')
    end
end

end

function chunk_ranges = get_chunk_ranges(z_start,z_end,max_chunk_size,chunk_pad,num_images)

% Get chunk z ranges
z_pos = z_start:z_end;
nchunks = 1;
chunk_z = length(z_pos);

if chunk_z < max_chunk_size
    chunk_start = max(z_start - chunk_pad,1);
    chunk_end = min(z_end + chunk_pad,num_images);

    chunk_start_adj = chunk_start;
    chunk_end_adj = chunk_end;
else
    while chunk_z > max_chunk_size
        nchunks = nchunks + 1;
        chunk_z = ceil(length(z_pos)/nchunks);
    end
    chunk_start = z_start:chunk_z:z_end;
    chunk_end = chunk_start + chunk_z;

    chunk_start(1) = z_start;
    chunk_end(end) = z_end;

    chunk_start_adj = chunk_start;
    chunk_start_adj(2:end) = chunk_start_adj(2:end) - chunk_pad;
    chunk_end_adj = chunk_end;
    chunk_end_adj(1:(nchunks-1)) = chunk_end_adj(1:(nchunks-1)) + chunk_pad;
end

chunk_ranges = [chunk_start' chunk_end' chunk_start_adj' chunk_end_adj'];
chunk_ranges(:,3) = max(chunk_ranges(:,3),1);
chunk_ranges(:,4) = min(chunk_ranges(:,4),num_images);

end

function seeds = get_seeds(I,I_max,scales,thresholds,rescale_resolution_factor)

seeds = ones(size(I));
% Detect cells from Hessian eigenvalues
for scale = scales
    %Sigma for given scale
    sigmas = scale*rescale_resolution_factor;

    % Apply 3D filter according to sigma
    I_filt = imGaussianFilter(I,round(sigmas*3),sigmas,'symmetric');
    I_filt = normalize_mins(I_filt) * I_max;
    
    % Compute Hessian coefficiants
    [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] = imHessian(I_filt, 0.9);
    
    % Compute eigenvalues
    %[l1,l2,l3] = imEigenValues3d(Dxx,Dyy,Dzz,Dxy,Dxz,Dyz);
    [l1,l2,l3]=eig3volume(Dxx,Dxy,Dxz,Dyy,Dyz,Dzz);

    % Keep voxels above threshold
    seeds = and(seeds, l1 < thresholds(1));
    seeds = and(seeds, l2 < thresholds(2));
    seeds = and(seeds, l3 < thresholds(3));
end

end

function seeds = cleanup_seeds(seeds)

%Closing
seeds = imclose(seeds, true(1, 1, 3));
seeds = imclose(seeds, true(1, 3, 1));
seeds = imclose(seeds, true(3, 1, 1));

%Opening
struc = false(3, 3, 3);
c = (3+1)/2;
struc(:, c, c) = true; struc(c, :, c) = true; struc(c, c, :) = true;
seeds = imopen(seeds, struc);

end

function centroids = kd_clean(centroids, radius)

% Create KD tree model
Mdl = KDTreeSearcher(centroids);
s = @(x) rangesearch(Mdl,x,radius,'Distance','euclidean');
jj = zeros(1,size(centroids,1));

% Find centroids within radius
for i = 1:size(centroids,1)
    jj(i) = cellfun(@length,s(centroids(i,:)));
end
idx_far = jj == 1;

% Take only first index for centroids that are too close
cen_sub = centroids(jj>1,:);
for i = 1:size(cen_sub)
    idx = cellfun(@min, s(cen_sub(i,:)));
    idx_far(idx) = 1;
end
centroids = centroids(idx_far,:);

end
