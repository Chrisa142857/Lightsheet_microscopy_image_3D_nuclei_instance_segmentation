function comp_img = blend_images(mov_img,ref_img,pseudo,blending_method,w)
%--------------------------------------------------------------------------
% Blend multi-tile images during stitching.
%--------------------------------------------------------------------------

% Option to create pseudocolored overlay
if nargin<3
    pseudo = true;
end

if ~pseudo && nargin<5
    error("Please provide blending method and weight vector")
end

% Calculate direction and overlapping regions by length and orientation of
% weight vector
[nrows, ncols] = size(ref_img);
if size(w,1) > size(w,2)
    direction = 'vertical';
    overlap_max = nrows-length(w)+1:nrows;
else
    direction = 'horizontal';
    overlap_max = ncols-length(w)+1:ncols;
end

if pseudo
    % Create pseudocolored overlay
    comp_img = create_overlay(mov_img, ref_img, overlap_max, direction);
else
    % Create fused, grayscle image
    overlap_min = 1:length(w);
    switch blending_method
        case {'sigmoid','linear'}
            inv_w = abs(1-w);
            if isequal(direction,'horizontal')
                %Perform non-linear weight
                ref_img(:,overlap_max) = ref_img(:,overlap_max).*inv_w +...
                    mov_img(:,overlap_min).*w;
                mov_img(:,overlap_min) = [];
            else
                %Perform non-linear weight
                ref_img(overlap_max,:) = ref_img(overlap_max,:).*inv_w +...
                    mov_img(overlap_min,:).*w;
                mov_img(overlap_min,:) = [];
            end
        case 'max'
            if isequal(direction,'horizontal')
                %Take max in the overlapping region
                ref_img(:,overlap_max) = max(ref_img(:,overlap_max),mov_img(:,overlap_min));
                mov_img(:,overlap_min) = [];
            else
                %Take max in the overlapping region
                ref_img(overlap_max,:) = max(ref_img(overlap_max,:),mov_img(overlap_min,:));
                mov_img(overlap_min,:) = [];
            end    
    end   
    % Concatenate images
    if isequal(direction,'horizontal')
        comp_img = horzcat(ref_img,mov_img);
    else
        comp_img = vertcat(ref_img,mov_img);
    end
end

end

function comp_img = create_overlay(mov_img,ref_img,overlap_max,direction)
% Create pseudocolor overlay

% Create reference space
[nrows, ncols] = size(mov_img);
r_ref_img = imref2d(size(ref_img));
r_mov_img = imref2d(size(mov_img));

% Adjust moving image position in world coordinates
if isequal(direction,'horizontal')
    r_mov_img.XWorldLimits(1) = min(overlap_max);
    r_mov_img.XWorldLimits(2) = min(overlap_max) + ncols;
else
    r_mov_img.YWorldLimits(1) = min(overlap_max);
    r_mov_img.YWorldLimits(2) = min(overlap_max) + nrows;
end

% Fuse
comp_img = imfuse(ref_img,r_ref_img,mov_img,r_mov_img,...
    'falsecolor','Scaling','joint','ColorChannels','red-cyan');

% Crop empty row or columns
if isequal(direction,'horizontal')
    comp_img = comp_img(:,any(sum(comp_img,3)>0,1),:);
else
    comp_img = comp_img(any(sum(comp_img,3)>0,2),:,:);
end

end