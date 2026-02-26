function L_new = expand_centroids3(L, I,rp)

res = size(I)./size(L);
L_new = zeros(size(I));

I_adj = imgaussfilt(I);
dim = size(L_new(:,:,1));
L_zero = zeros(dim);

centroids_full = arrayfun(@(s) s.Centroid.*res,rp,'UniformOutput',false); 
centroids_full = cell2mat(centroids_full);
centroids_full(:,4) = ceil(centroids_full(:,3)-0.25);

% Set intensity and distance thresholds determine expansion
background = double(I_adj(I_adj<otsuthresh(I_adj(:))*65535));
%int_thresh = median(background) + 3*std2(background(:));
int_thresh = 0;

a = 1;
for i = unique(centroids_full(:,3))'
    centroids_subset = centroids_full(centroids_full(:,3) == i,:);
    z_pos = centroids_subset(1,4);
    adj_img = I_adj(:,:,z_pos);
    if z_pos+1 < size(L_new,3) && z_pos-1 > 0 && i-z_pos < 0 
        adj_img2 = I_adj(:,:,z_pos-1);
        adj = -1;
    elseif z_pos+1 < size(L_new,3) && z_pos-1 > 0 && i-z_pos > 0
        adj_img2 = I_adj(:,:,z_pos+1);
        adj = 1;
    else
        adj_img2 = adj_img;
        adj = 0;
    end

    for j = 1:size(centroids_subset)
        L_slice = L_zero;
        pos = centroids_subset(j,1:2);

        y_range = floor(pos(1)):ceil(pos(1));
        y_range(y_range==0) = 1;
        y_range(y_range>size(L_new,1)) = size(L_new,1);

        x_range = floor(pos(2)):ceil(pos(2));
        x_range(x_range==0) = 1;
        x_range(x_range>size(L_new,2)) = size(L_new,2);
        
        L_slice(x_range, y_range) = 1;
        intensity = mean(adj_img(L_slice == 1));
    
        % If centroids falls between two z positions becausing of
        % downsizing, use intensity to decide where to place centroid
        if abs(z_pos-i) > 0.25
            intensity2 = mean(adj_img2(L_slice == 1)); 
            if intensity2 > intensity
               z_pos_adj = z_pos + adj;
               intensity = intensity2;
            else
               z_pos_adj = z_pos;
            end
        else
            z_pos_adj = z_pos;
        end
        
        % Expand dim cells
        if intensity < int_thresh
            pos = round(centroids_subset(j,1:2));        
        
            y_range = pos(1)-1:pos(1)+1;
            y_range(y_range==0) = 1;
            y_range(y_range>size(L_new,1)) = size(L_new,1);

            x_range = pos(2)-1:pos(2)+1;
            x_range(x_range==0) = 1;
            x_range(x_range>size(L_new,2)) = size(L_new,2);

            L_slice(x_range, y_range) = 1;
            a = a+1;
        end
        
        L_new(:,:,z_pos_adj) = L_new(:,:,z_pos_adj) + L_slice;
        centroids_full(a,5) = z_pos_adj;
        a = a+1;
    end
end        

end