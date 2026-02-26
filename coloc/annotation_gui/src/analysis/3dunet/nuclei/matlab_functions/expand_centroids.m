function L_new = expand_centroids(L, I, se_norm, se_small, se_big, thresh)

%labels = unique(L(:));
L_new = zeros(size(L));
dim = size(L(:,:,1));
n = 1;

I = imresize3(I,size(L));
int_thresh = median(double(I)) + std(double(I(:)))*2;

for i = 1:size(L,3)
    pos = [];
    L_slice = L(:,:,i);
    I_slice = I(:,:,i);
    [pos(:,1),pos(:,2)] = ind2sub(dim,find(L_slice>0));
    labels = unique(L_slice(:));
    
    for j = 2:length(labels)
        L_empty1 = zeros(dim);
        L_empty2 = zeros(dim);
        % Position of predicted centroid
        [pos1(1),pos1(2)] = ind2sub(dim,find(L_slice==labels(j)));
        % Measure distance from every true centroid
        D = sqrt(sum((pos - pos1).^2, 2));
    
        nearest = min(D(D>0));
        L_empty1(pos1(1),pos1(2)) = n;
        L_empty2(pos1(1),pos1(2)) = n;

        
        if nearest > thresh
            % Expand more if no centroids nearby
            L_empty1 = imdilate(L_empty1,se_small);
            if mean(I_slice(logical(L_empty1))) < int_thresh
                L_empty2 = imdilate(L_empty2,se_big);
            else
                L_empty2 = L_empty1;
            end
        elseif nearest < (thresh/2)+0.5
            % Expand less if centroids nearby
            L_empty2 = imdilate(L_empty1,se_norm);
        else
            L_empty2 = imdilate(L_empty1,se_small);
        end
        L_new(:,:,i) = L_new(:,:,i) + L_empty2;
        n = n+1;
    end 
end
end