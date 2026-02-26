function L_new = expand_centroids2(L, I, n_pix)

% Set intensity and distance thresholds determine expansion
background = double(I(I<otsuthresh(I(:))*65535));
int_thresh = median(background) + 2*std2(background);
dist_thresh = 5;

% Create structuring objects
nhood1 = [1 1 1;1 1 1;1 1 1];
nhood2 = [0 1 1 1 0; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 0 1 1 1 0];
se1 = strel(nhood1);
se2 = strel(nhood2);

pix_min = n_pix;
pix_max = sum(nhood1(:));

n = linspace(int_thresh,65535,pix_max-pix_min+2);
n = n(2:end-1);
nhood_n = fliplr(pix_min:pix_max);


% Resize image to match centroid image size
I = imresize3(I,size(L));

% Dimension 
dim = size(L(:,:,1));
L_new = zeros(size(L));

a=0;
b=0;
% Perform centroind expansion seperately for each slice
for i = 1:size(L,3)
    L_slice = L(:,:,i);
    I_slice = I(:,:,i);
    % Get positions and labels for each centroid in the slice
    [x,y] = ind2sub(dim,find(L_slice>0));
    labels = unique(L_slice(:));
    % Adjust for each centroid
    for j = 2:length(labels)
        L_empty = zeros(dim);
        nhood_eroded = nhood1;
        % Position of predicted centroid
        [x1,y1] = ind2sub(dim,find(L_slice==labels(j)));
        % Measure distance from every true centroid
        D = sqrt(sum(([x,y] - [x1,y1]).^2, 2));
        nearest = min(D(D>0));
        % Create intial expanded box
        L_empty(x1,y1) = 1;
        L_empty1 = imdilate(L_empty,se1);
        box_int = double(I_slice(logical(L_empty1)));
        if median(box_int) < int_thresh && nearest > dist_thresh
            % Large cells with low intensity
            L_empty2 = imdilate(L_empty,se2);
            a = a + 1;
        else
            % Small cells get eroded neighborhood scaled by mean intensity
            cen_size = min(nhood_n(mean(box_int)' > n))-1;
            [~, pix_idx] = sort(box_int,'descend');
            % Adjust for cells along the edge
            if length(box_int) < 9
                cen_size = round(cen_size*length(box_int)/9);
                pix_idx = pix_idx(1:cen_size);
                nhood_eroded(pix_idx(cen_size+1:end)) = 0;

            elseif cen_size == 5
                nhood_eroded = [0 1 0;1 1 1;0 1 0];            
            elseif cen_size <9
                edge_idx = [1,3,7,9];
                box_int(edge_idx) = box_int(edge_idx)/2;
                [~, pix_idx] = sort(box_int,'descend');

                nhood_eroded(pix_idx(cen_size+1:end)) = 0;
            else
                nhood_eroded = nhood1;
            end
            L_empty2 = imdilate(L_empty,strel(nhood_eroded));
            L_empty2(x1,y1) = 1;
            %disp(sum(nhood_eroded(:)))
            b = b +1;
        end
        % Add expanded centroids to new image
        L_new(:,:,i) = L_new(:,:,i) + L_empty2;
    end 
end
%imshowpair(imadjust(I(:,:,29)),L_new(:,:,29))
end