function annotations = reannotate_centroids(centroids,config)
% Load mask and reannotate centroids 

I_mask = load_var(config,strcat(config.sample_id,'_mask'),'I_mask');
s = config.resolution{1}/config.resample_resolution;

% Resize mask
[mrows,mcols,mslices] = size(I_mask);

% Adjust for rounding
yxz = round(centroids(:,1:3).*s(1:3));
yxz(yxz == 0) = 1;
yxz(:,1) = min(yxz(:,1),mrows);
yxz(:,2) = min(yxz(:,2),mcols);
yxz(:,3) = min(yxz(:,3),mslices);

% Get annotation at linear index
yxz = sub2ind([mrows,mcols,mslices],yxz(:,1),yxz(:,2),yxz(:,3));
annotations = double(I_mask(yxz));

end






