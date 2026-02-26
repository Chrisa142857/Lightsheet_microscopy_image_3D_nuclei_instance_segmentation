function normalize_stack_intensity(path_table, ranges, overwrite, method, crop_range)
%--------------------------------------------------------------------------
% Normalize intensity variation along Z using semi-quantile normalization.
%--------------------------------------------------------------------------
% Usage: 
% I_adj = normalize_stack_intensity(stack, ranges, overwrite, method, crop_range)
% 
% Inputs:
% path_table: (table) Table containg file images to normalize.
%
% ranges: (1x4 cell array) Row, column, z-range, & channel to normalize. 
% For example, {1,1,100:200, 1} normalizes tile position 01x01 for slices 
% 100 to 200 in channel number 1. Leave index empty ([]) to run on all
% positions of that range.
%
% overwrite: (logical) Replace raw images with adjusted. Otherwise adjusted
% images will be saved in new folder. (default: false)
%
% method: ('contrast','quantile') Use contrast stretching or semi-quantile
% normalization method. (default: 'contrast')
%
% crop_range: [row_start, row_end, col_start, col_end] Crop to this ROI for
% measuring thresholds. (default: empty - use full image)
% 
% Ouputs:
% I_adj: Stack of adjusted images.
%--------------------------------------------------------------------------

if nargin<2
    ranges{1} = unique(path_table.y);
    ranges{2} = unique(path_table.x);
    ranges{3} = unique(path_table.z);
    ranges{4} = unique(path_table.channel_num);
end
    
if nargin<3
    overwrite = false;
end

if nargin<4
    method = 'contrast';
end

if length(ranges)<4
    ranges = cat(2,ranges,cell(1,4-length(ranges)));
end

if nargin<5
    crop_range = [];
end

if isempty(ranges{1}); ranges{1} = unique(path_table.y); end
if isempty(ranges{2}); ranges{2} = unique(path_table.x); end
if isempty(ranges{3}); ranges{3} = unique(path_table.z); end
if isempty(ranges{4}); ranges{4} = unique(path_table.channel_num); end

% Keep only images in range
z = ranges{3};
path_table = path_table(ismember(path_table.z,z),:);

% Perform normalization across multiple tile positions
y = ranges{1};
x = ranges{2};
c_idx = ranges{4};
disp(y)
disp(x)
disp(c_idx)

for i = 1:length(x)
    for j = 1:length(y)
        for k = 1:length(c_idx)
            sub_stack = path_table(path_table.x == x(i) & path_table.y == y(j) &...
                path_table.channel_num == c_idx(k),:);
            if isempty(sub_stack)
                continue
            else
                normalize_tile_stack(sub_stack,overwrite,method,crop_range)
            end
        end
    end
end

end


function normalize_tile_stack(stack,overwrite,method,crop_range)

% Get image paths
files = stack.file;
[~,names,ext] = cellfun(@(s) fileparts(s),stack.file,'UniformOutput',false);
names = cellfun(@(s,t) string(s)+string(t),names,ext);

% Read images
tempI = imread(files{1});
[nrows, ncols] = size(tempI);
I = zeros(nrows,ncols,length(files),'uint16');
fprintf("Reading images... \n")
for i = 1:length(files)
    I(:,:,i) = imread(files{i});
end

if isequal(method,'contrast')
    fprintf("Normalizing stack using cotrast stretching. \n")
    contrast_stretch(I,files,names,overwrite,crop_range)
else
    fprintf("Performing semi-quantile normalization on image stack. \n")
    semi_quantile(I,filesnames,overwrite)
end

end

function contrast_stretch(I,files,names,overwrite,crop_range)

% Get thresholds for first and last slice
fprintf("Calculating threshold adjustments... \n")

[nrows, ncols] = size(I);
if isempty(crop_range)
    rows = 1:nrows;
    cols = 1:ncols;
end

% Get threshold percentiles for first and last slice
img1 = smooth_background_subtraction(I(rows,cols,1),true);
bw = imbinarize(imadjust(img1));
high_prct(1) = (1-sum(bw(:))/numel(bw(:)))*100;
low_prct(1) = single(median(img1(:)));

img1 = smooth_background_subtraction(I(rows,cols,end),true);
bw = imbinarize(imadjust(img1));
high_prct(2) = (1-sum(bw(:))/numel(bw(:)))*100;
low_prct(2) = single(median(img1(:)));

% Interpolate percentiles across range
high_prct_int = interp1([1,length(files)],[high_prct(1), high_prct(2)], 1:length(files));
low_prct_int = interp1([1,length(files)],[low_prct(1), low_prct(2)], 1:length(files));

% Get actual intensities at percentiles
s1 = single(I(rows,cols,1));
s2 = single(I(rows,cols,end));
max_int1 = prctile(s1(:),high_prct_int(1));
max_int2 = prctile(s2(:),high_prct_int(end));

min_int1 = prctile(s1(:),low_prct_int(1));
min_int2 = prctile(s2(:),low_prct_int(end));

% Interpolate expected intensities in between
max_interp = interp1([1,length(files)],[max_int1, max_int2], 1:length(files));
min_interp = interp1([1,length(files)],[min_int1, min_int2], 1:length(files));

% Caculate adjustment above upper percentile for each slice
x1 = ones(1,length(files));
x2 = ones(1,length(files));
for i = 1:length(files)
    s2 = single(I(:,:,i));        
    
    % Bring low intensities
    min_int2 = prctile(s2(:),low_prct_int(i));
    x1(i) = min_interp(i)/min_int2;
    s2 = s2*x1(i);
    
    % Bring up high intensities
    max_int2 = prctile(s2(:),high_prct_int(i));
    x2(i) = (max_interp(i)-min_interp(i))/(max_int2-min_interp(i));
end
% Smooth results
x1 = smooth(x1,9,'sgolay');
x2 = smooth(x2,9,'sgolay');

% Create directory
save_dir = fullfile(pwd,'adjusted');
if exist(save_dir,'dir') ~= 7
    mkdir(save_dir);
end

% Apply adjustments
fprintf("Applying adjustments... \n")
I_adj = uint16(I);
for i = fliplr(1:length(files))
    s = single(I(:,:,i));
    s = s*x1(i);
    s = (s-min_interp(i))*x2(i) + min_interp(i);
    
    % Save back into matrix
    I_adj(:,:,i) = uint16(s);
    
    % Convert to uint16 and write images
    if overwrite
        % Save into original directory
        filename = stack.file(i);
        imwrite(I_adj(:,:,i),filename)
    else
        % Write files
        filename = fullfile(save_dir, names(i));
        imwrite(I_adj(:,:,i),filename)
    end
end

end


function semi_quantile(I,files,names,overwrite)

% Get thresholds for first and last slice
fprintf("Calculating threshold adjustments... \n")

% Get threshold percentiles for first and last slice
img1 = smooth_background_subtraction(I(:,:,1),true);
bw = imbinarize(imadjust(img1));
high_prct(1) = (1-sum(bw(:))/numel(bw(:)))*100;

img1 = smooth_background_subtraction(I(:,:,end),true);
bw = imbinarize(imadjust(img1));
high_prct(2) = (1-sum(bw(:))/numel(bw(:)))*100;

% Interpolate percentiles across range
high_prct_int = interp1([1,length(files)],[high_prct(1), high_prct(2)], 1:length(files));

% Get actual intensities at percentiles
s1 = single(I(:,:,1));
s2 = single(I(:,:,end));
max_int1 = prctile(s1(:),high_prct_int(1));
max_int2 = prctile(s2(:),high_prct_int(end));

% Interpolate expected intensities in between
max_int1 = interp1([1,length(files)],[max_int1, max_int2], 1:length(files));

% Caculate adjustment above upper percentile for each slice
x2 = ones(1,length(files));
for i = 1:length(files)
    s2 = single(I(:,:,i));        
    % Bring up high intensities
    max_int2 = prctile(s2(:),high_prct_int(i));
    x2(i) = max_int1(i)/max_int2;
end
% Smooth results
x2 = smooth(x2,9,'sgolay');

% Prep quantile part
% Get ordered vectors
fprintf("Calculating interpolated pixel quantiles... \n")
img1 = sort(s1(:));
img2 = sort(s2(:));

% Subset only pixels below upper percentile
idxs = round((max(high_prct_int)/100)*length(img1));
img1 = img1(1:idxs);
img2 = img2(1:idxs);

% Interpolate all expected pixels for each slice
[a,~,c] = unique([img1,img2],'rows');
v = arrayfun(@(s,t) interp1([1,length(files)],[s, t],1:length(files)),a(:,1),a(:,2),'UniformOutput',false);
v = cat(1,v{:});
v = v(c,:);

% Create directory
save_dir = fullfile(pwd,'adjusted');
if exist(save_dir,'dir') ~= 7
    mkdir(save_dir);
end

% Apply adjustments
fprintf("Applying adjustments... \n")
I_adj = uint16(I);
for i = fliplr(1:length(files))
    s = single(I(:,:,i));
    DoS = std2(s(s<median(s(:))));
    max_int2 = prctile(s(:),high_prct_int(i));
    quant_idxs = length(s(s<max_int2));

    % Contrast stretch
    s(s>=max_int2) = s(s>=max_int2)*x2(i);

    % Quantile normalization
    [~,idx] = sort(s(:));
    idx = idx(1:quant_idxs);
    s(idx) = v(1:quant_idxs,i);

    % Apply non-local means smoothing. Degree of smoothing scaled 
    s = reshape(s, size(I(:,:,i)));
    s = single(imnlmfilt(s,'ComparisonWindowSize',3,'DegreeOfSmoothing',DoS*x2(i)));
    
    % Save back into matrix
    I_adj(:,:,i) = uint16(s);
    
    % Convert to uint16 and write images
    if overwrite
        % Save into original directory
        filename = stack.file(i);
        imwrite(I_adj(:,:,i),filename)
    else

        % Write files
        filename = fullfile(save_dir, names(i));
        imwrite(I_adj(:,:,i),filename)
    end
end
end