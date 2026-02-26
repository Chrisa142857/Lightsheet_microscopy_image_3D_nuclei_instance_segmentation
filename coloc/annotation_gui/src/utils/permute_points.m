function [pts_out,u_idx] = permute_points(pts_in, img_size, or_in, or_out, linear_idx, keep_unique)
%--------------------------------------------------------------------------
% Permute orientation of a 3D point set. See permute_orientation.
%
% Orientation key: 
% anterior(a)/posterior(p), 
% superior(s)/inferior(i), 
% left(l)/right(r)
%--------------------------------------------------------------------------
% Usage:
% pts_out = permute_points(pts_in, img_size, or_in, or_out)
%
%--------------------------------------------------------------------------
% Inputs:
% pts_in: (n x 3 int) Point coordinates in old orientation. Specified as
% row(y), column(x), slice(z).
%
% img_size: (numeric) Size of image from which points are sampled. Can also
% just provide image itself.
%
% or_in: (1x3 char) Input image orientation.
%
% or_out: (1x3 char) Output image orientation.
%
% linear_idx: Whether input are linear indexes (default: false)
% 
%--------------------------------------------------------------------------
% Outputs:
% pts_out: (n x 3 int) Point coordinates in new orientation.
%
%--------------------------------------------------------------------------

if nargin<5
    linear_idx = false;
end

if nargin<6
    keep_unique = false;
end

% Check orientation characters
if isstring(or_in)
    or_in = char(or_in);
end
if isstring(or_out)
    or_out = char(or_out);
end

assert(length(or_in) == 3, "Input/output orientation should be 1x3 character arrays")
assert(length(or_out) == 3, "Input/output orientation should be 1x3 character arrays")

% Check image size
if ndims(img_size) == 3
    img_size = size(img_size);
end

% Check if input/output are the same
if string(or_in) == string(or_out)
    pts_out = pts_in;
    return
end

% Scan characters to see which to permute
yxz_in = 1:3;
for i = 1:3
    if ismember({or_out(i)},{'a','p'})
        yxz_in(i) = find(or_in == 'a' | or_in == 'p');
        if ismember({or_in(3)},{'a','p'})
            idx = i; 
            s = 'ap';
        end
    elseif ismember({or_out(i)},{'i','s'})
        yxz_in(i) = find(or_in == 'i' | or_in == 's');
        if ismember({or_in(3)},{'i','s'}) 
            idx = i; 
            s = 'is';
        end
    elseif ismember({or_out(i)},{'l','r'})
        yxz_in(i) = find(or_in == 'l' | or_in == 'r');
        if ismember({or_in(3)},{'l','r'})
            idx = i; 
            s = 'lr';
        end
    else
        error("Unrecognized orientation character specified")
    end
end
assert(length(unique(yxz_in)) == 3,"All 3 axes not specified correctly")

% Permute
pts_out = pts_in(:,yxz_in);

% Flip if opposite axis
idx = 0;
for i = 1:3
    if or_in(yxz_in(i)) ~= or_out(i)
        pts_out(:,i) = img_size(i)-pts_out(:,i)+1;
        idx = idx+1;
    end
end

% Keep ony unique coordinates
if keep_unique
    [pts_out,~,u_idx] = unique(pts_out,'rows');
end

% Convert points to linear index
if linear_idx
    img = zeros(img_size);
    img = permute_orientation(img,or_in,or_out);
    pts_out = sub2ind(size(img),pts_out(:,1),pts_out(:,2),pts_out(:,3));
end

end
