function labels_trimmed = trim_labels_around_edges(labels)




rp = regionprops(label);
cen_truth = rp.Centroid; 


cen_truth = reshape([rp.Centroid],[3,length(rp)])';

end