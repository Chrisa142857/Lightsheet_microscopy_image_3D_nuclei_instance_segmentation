function img = permute_orientation(img, or_in, or_out, hemisphere)
%--------------------------------------------------------------------------
% Permute sample anotomical orientation. Orientations are specified as 1x3 
% character vectors. Row, column, slice at idx=1 in the image should match 
% 1st, 2nd, 3rd characters in each vector in this order.
%
% Orientation key: 
% anterior(a)/posterior(p), 
% superior(s)/inferior(i), 
% left(l)/right(r)
%--------------------------------------------------------------------------
% Usage:
% img = permute_orientation(img, or_in, or_out)
%
% Example:
% img = permute_orientation(img,'rsa','ail');
% Permutes image from 'right','superior','anterior' at (y=1,x=1,z=1) to
% 'anterior','inferior','left'
%--------------------------------------------------------------------------
% Inputs:
% img: (numeric) Input 3D image. 
%
% or_in: (1x3 char) Input image orientation.
%
% or_out: (1x3 char) Output image orientation.
%--------------------------------------------------------------------------

% Check for full brain
if nargin<4
    hemisphere = true;
end

%if ~hemisphere
% 'rsa','sar','ars','rip','pri','ipr'
% 'lsp','pls','sla','las','ali','pri','srp','ial','ipr'

% Check orientation characters
if isstring(or_in)
    or_in = char(or_in);
end
if isstring(or_out)
    or_out = char(or_out);
end

assert(length(or_in) == 3, "Input/output orientation should be 1x3 character arrays")
assert(length(or_out) == 3, "Input/output orientation should be 1x3 character arrays")

% Check if input/output are the same
if string(or_in) == string(or_out)
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
if length(size(img)) == 4
    yxz_in = [yxz_in,4];
end
img = permute(img,yxz_in);

% Flip last axis
if ismember(or_in(idx),s)
    %img = flip(img,idx);
end

% Flip if opposite axis
idx = 0;
for i = 1:3
    if or_in(yxz_in(i)) ~= or_out(i)
        img = flip(img,i);
        idx = idx+1;
    end
    if idx == 3
        %img = flip(img,3);
    end
end

end
