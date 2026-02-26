function final_img = create_performance_image(label,image,centroids, mult_lbl,...
    miss_lbl, merge_lbl)
% Create image indicating cell detection mistakes
% Binarize label matrix
labels = unique(label);
n_labels = length(labels);
new_labels = zeros(size(label));

%label = imrotate(label,90);
%label = flipdim(label,1); 

for i = 1:size(centroids,1)
    pos = centroids(i,:);
    lbl = double(label(pos(1),pos(2),pos(3)));
    % Multi
    if intersect(lbl,mult_lbl(:,3))       
        new_labels(pos(1),pos(2),pos(3)) = 2;
    % Correct
    else
        new_labels(pos(1),pos(2),pos(3)) = 1;
    end
end

rp = regionprops(double(label));

% Add missing
for i = 1:size(rp,1)
    pos = round(rp(i).Centroid);
    if ~isnan(pos(1))
        val = double(label(pos(1),pos(2),pos(3)));
        if intersect(val,miss_lbl(:,3))  
            new_labels(pos(1),pos(2),pos(3)) = 3;
        elseif intersect(val,merge_lbl(:,3))  
            new_labels(pos(1),pos(2),pos(3)) = 3;
        end
    end
end

% Trim label images to area with labels
left = min(find(max(sum(label,3)) ~= 0));
right = max(find(max(sum(label,3)) ~= 0));
top = min(find(max(sum(label,2)) ~= 0));
bottom = max(find(max(sum(label,2)) ~= 0));

new_labels = new_labels(left:right,left:right,top:bottom);
new_image = image(left:right,left:right,top:bottom);


new_image = imadjustn(new_image);

se = strel([0 1 0;1 1 1;0 1 0]);
new_labels = imdilate(new_labels,se);

map = [1 0 0;
    0 1 0;
    0 0 1];

for i = 1:size(new_labels,3)
    img(:,:,i,:) = label2rgb(new_labels(:,:,i),map,'k');
    final_img(:,:,i,:) = imfuse(new_image(:,:,i),squeeze(img(:,:,i,:)),...
        'blend','Scaling','independent');
end

a = 1;

%for i = 2:n_labels
%    current_label = labels(i);
%    idx = find(label==current_label);
    
    
    
%    idx = idx(prediction(idx) > 0);
    
 %   if ismember(current_label,mult_lbl(:,3))
 %       new_labels(idx) = 4;
 %   elseif ismember(current_label,miss_lbl(:,3))
%        new_labels(idx) = 3;
%    elseif ismember(current_label,merge_lbl(:,3))
%        new_labels(idx) = 2;
%    else
%        new_labels(idx) = 1;
%    end
%end
    

%map = [1 1 1;
%    1 0 0;
%    0 1 0;
%    0 0 1];



%img = flipdim(img,1);
%img = imrotate(img,-90);

end