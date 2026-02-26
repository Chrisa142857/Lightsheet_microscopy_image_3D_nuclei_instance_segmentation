function I_adj = dog_adjust(I,nuc_radius,minmax,weight)
%--------------------------------------------------------------------------
% Enhacen blob objects using Difference of Gaussian filter
%--------------------------------------------------------------------------
%
% Usage:
% I_adj = dog_adjust(I,nuc_radius,minmax,weight)
%
% Inputs:
% I - input 2D image of any type.
%
% nuc_radius - average pixel radius of blobs.
%
% minmax - (default: [0.75,4]) 2 element vector specify separation between
% small and large gaussians. Keep around 1.
%
% weight - (default: 1) weight od DoG image. For example, if 0.5, takes
% average of DoG and original images.
%
% Outputs:
% I_adj - DoG adjusted image.
%--------------------------------------------------------------------------

if nargin <3
    minmax = [0.75,4];
end

if nargin < 4
    weight = 1;
end

% Calculate filter size based on min/max values around some blob radius.
% Use factor to adjust
factor = 0.15;
s = factor*([nuc_radius*minmax(1) nuc_radius*minmax(2)]);
filter_size = 2*ceil(s(2))+1;

%Check to see if single
img_class = class(I);
if ~isequal(img_class,'single')
    I = single(I);
end

% Instead of subtracting small from large and detecting edges, subtract
% large from small to detect blobs
I2 = imgaussfilt(I,s(1),'FilterSize',filter_size);
I3 = imgaussfilt(I,s(2),'FilterSize',filter_size);
dog = I2-I3;

% Adjusted image the mean of original image and DoG image scaled by some
% weight
I_adj = I*(1-weight) + dog*weight;
scale = (max(I(:))-min(I(:)))/(max(I_adj(:))-min(I_adj(:)));
I_adj = I_adj*scale;
I_adj(I_adj<0) = 0;
%imshow(imadjust(uint16(I_adj)))

% Recast to original class if necessary
if ~isequal(I_adj, img_class)
    I_adj = cast(I_adj,img_class);
end

end