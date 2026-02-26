function [back_thresh, std_thresh] = get_background_at_centroids(centroids, config, B, S)

% Convert centroid positions to background image space
cen_location = centroids(:,1:3).*config.resolution;
cen_mask = round(cen_location./config.back_res);
res_dims = size(B);

% Limit any indexes that fall out of range of the background image
% after round
cen_mask(any(cen_mask==0)) = 1;
cen_mask(cen_mask(:,1) > res_dims(1)) = res_dims(1);
cen_mask(cen_mask(:,2) > res_dims(2)) = res_dims(2);
cen_mask(cen_mask(:,3) > res_dims(3)) = res_dims(3);

% Get linear indexes for centroid positions in the background image
cen_idx = sub2ind(res_dims, cen_mask(:,1),cen_mask(:,2), cen_mask(:,3));
        
% Append values for background and standard deviation
back_thresh = single(B(cen_idx));
std_thresh = single(S(cen_idx));
end
