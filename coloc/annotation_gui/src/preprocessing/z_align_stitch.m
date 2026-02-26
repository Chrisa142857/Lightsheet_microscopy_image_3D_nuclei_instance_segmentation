function [z_displacement,q,low_flag] = z_align_stitch(path_mov,path_ref,overlap,z_positions,z_window,direction,signalThresh)
%--------------------------------------------------------------------------
% Determine pairwise z displacement between z-stack tiles by taking
% sections within a moving image and performing 2D phase correlation to the
% overlapped region with the reference stack. Average cross correlation for
% the selected slices in the stack are then used to determine the final
% z-displacement.
%--------------------------------------------------------------------------

% Defaults
peaks = 3;
usfac = 1;
min_overlap = 50;
min_signal = 0.025;       % Minimum fraction of signal pixels for registering
z_window_narrow = 1;

% Adjust signalThresh to 16 bit
if signalThresh<1
    signalThresh = signalThresh*65535*2;
end

% Check file lengths
nfiles_mov = height(path_mov);    
nfiles_ref = height(path_ref);
if nfiles_mov ~= nfiles_ref
    error("Number of images do not match up")
end

% Adjust z_positions if specified number is too high
if z_positions<1
    z_positions = ceil(z_positions*nfiles_ref);
end
z_positions = min(z_positions,nfiles_mov-z_window);

% Pick reference images in range
z = round(linspace(z_window+1,nfiles_ref-z_window,z_positions));
path_ref = path_ref(z,:);

% Determine overlap region
ref_img = imread(path_ref.file{1});
[nrows, ncols] = size(ref_img);

if direction == 1 && ncols*overlap < min_overlap
    overlap = min(1,min_overlap/ncols);
elseif direction == 1 && nrows*overlap < min_overlap
    overlap = min(1,min_overlap/nrows);
end

if direction == 1
    %Horizontal image overlap regions
    overlap_min = {[1,nrows],[1,round(ncols*overlap)]};
    overlap_max = {[1,nrows],[ncols-overlap_min{2}(2)+1,ncols]};
else
    %Vertical image overlap regions
    overlap_min = {[1,round(nrows*overlap)],[1,ncols]};
    overlap_max = {[nrows-overlap_min{1}(2)+1,nrows],[1,ncols]};
end

% Load reference images to match with
f = zeros(1,length(z_positions));
ref_img = zeros(overlap_max{1}(2)-overlap_max{1}(1)+1,...
    overlap_max{2}(2)-overlap_max{2}(1)+1,...
    z_positions,'uint16');
for i = 1:z_positions    
    img = imread(path_ref.file{i},'PixelRegion',overlap_max);
    % Measure number of positive pixels
    f(i) = round(numel(img(img>signalThresh))/numel(img),3);    
    ref_img(:,:,i) = img;
end

% Check reference images to see if there are bright pixels
if ~any(f>min_signal)
    %Save tranform
    warning("Low intensities measured for all z positions sampled")
    z_displacement = NaN;
    q = -1;
    low_flag = 1;
    return
end

% Find z position with most positive pixels
[~,idx] = max(f);
z_initial = z(idx);

% Check cross correlation for wide range at this z position
z_sub_range = z_initial-z_window:1:z_initial+z_window;
ref_img0 = ref_img(:,:,idx);
cc0 = zeros(1,length(z_sub_range));
for i = 1:length(z_sub_range)
    % Read corresponding moving image
    mov_img = imread(path_mov.file{z_sub_range(i)},'PixelRegion',overlap_min);
    % Get transform using phase correlation and measure cross correlation
    [~,~,~,~,cc0(i)] = calculate_phase_correlation(mov_img,ref_img0,peaks,usfac);
end

% Find displacement with highest cross correlation
[~,idx] = max(cc0);
z_shift = idx-z_window-1;

% Use narrower range test for remaining z_positions
cc = zeros(z_positions,z_window_narrow*2+1);
for i = 1:length(z)
    z_sub_range = z(i)-z_window_narrow+z_shift:1:z(i)+z_window_narrow+z_shift;
    for j = 1:length(z_sub_range)
        if z_sub_range(j) < 1 || z_sub_range(j) > nfiles_mov
            continue
        else
            % Read corresponding moving image
            mov_img = imread(path_mov.file{z_sub_range(j)},'PixelRegion',overlap_min);
            % Get transform using phase correlation and measure cross correlation
            [~,~,~,~,cc(i,j)] = calculate_phase_correlation(mov_img,ref_img(:,:,i),peaks,usfac);
        end
    end
end

% Find displacement with highest cross correlation
[q1,idx] = max(sum(cc,1));
z_shift_narrow = idx-z_window_narrow-1;

% Calculate final displacement and quality metric
z_displacement = z_shift + z_shift_narrow;

mean_cc = mean(cc(:,idx));
cc(:,idx) = 0;
q = (q1-max(sum(cc,1)));
low_flag = 0;

% Display average cross correlation
fprintf("\t\t\t Z displacement: %d \t Mean cross correlation: %.3f \n",z_displacement,mean_cc);
%disp(z_initial)
end
