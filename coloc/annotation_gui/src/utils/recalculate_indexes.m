function cen_idx = recalculate_indexes(centroids, cen_sub)
pos = cen_sub(:,1:3);
cen_idx = zeros(size(cen_sub,1),1);
for i = 1:size(cen_sub,1)
    cen_idx(i) = find(centroids(:,1) == pos(i,1) & centroids(:,2) == pos(i,2) &...
        centroids(:,3) == pos(i,3));
end
end