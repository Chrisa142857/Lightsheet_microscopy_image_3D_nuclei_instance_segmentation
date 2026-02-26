function v_out = rescale_ind(v,img_in,res_in,res_out)
%--------------------------------------------------------------------------
% Rescale linear indexes between different resolutions.
%
% Note: indexes falling out of new dimensions are trimmed.
%--------------------------------------------------------------------------
% Usage:
% v_out = rescale_ind(v,img_in,res_in,res_out)
%
%--------------------------------------------------------------------------
% Inputs:
% v: (int) Vector of linear indexes.
%
% img_in: (string) Dimensions of input 3D image from which indexes are
% sampled. Or 3D image itself. 
% 
% res_in: Input resolution of indexes.
%
% res_out: Target resolution of indexes. 
%
%--------------------------------------------------------------------------
% Outputs:
% v_out: (int) Rescald linear indexes.
%
%--------------------------------------------------------------------------

if size(v,1)<size(v,2)
    v = v';
end

if ndims(img_in) == 3
    img_size = size(img_in);
end

% Rescale linear indexes from 1 resolution to another
[y,x,z] = ind2sub(img_size,v);
s = [y,x,z];
res_adj = res_in./res_out;
s = round(s.*res_adj);
new_dims = round(img_size.*res_adj);

% Round to int, trim, and convert back to linear indexes
if all(s == 0,'all')
    v_out = [];
    return
else
    %s = s(all(s>1,2),:);    
end

if isempty(s)
    v_out = [];
    return
else
    %s = s(all(s(:,1)<=new_dims(1),2),:);
    %s = s(all(s(:,2)<=new_dims(2),2),:);
    %s = s(all(s(:,3)<=new_dims(3),2),:);
end
v_out = sub2ind(new_dims,s(:,1),s(:,2),s(:,3));

end