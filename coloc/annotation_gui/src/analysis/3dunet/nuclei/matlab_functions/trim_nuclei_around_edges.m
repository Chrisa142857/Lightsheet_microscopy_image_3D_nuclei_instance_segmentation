function labels= trim_nuclei_around_edges(labels,padding)

labels_pre_trimmed = length(unique(labels(:)));

% Get region props for all unique labels
rp = regionprops(labels);

top_left = padding;
bottom_right = size(labels) - padding;

% Remove labels whose centroids falls around the outer edges defined by
% padding
for i = 1:length(rp)
    if rp(i).Area > 0 
        pos = rp(i).Centroid;
        
        if pos(1) < top_left(1) || pos(1) > bottom_right(1) || pos(2) < top_left(2) ||...
                pos(2) > bottom_right(2) || pos(3) < top_left(3) || pos(3) > bottom_right(3)
            
            labels(labels == i) = -1;
        end
    end
end

% Display how many nuclei have been trimmed
labels_post_trimmed = length(unique(labels(:)));
fprintf('\n Trimmed %d nuclei\n',labels_pre_trimmed - labels_post_trimmed)
end