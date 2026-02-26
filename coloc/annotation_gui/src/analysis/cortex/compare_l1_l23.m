function p = compare_l1_l23(centroids)

l_table = readtable(fullfile('annotations','cortical_layers',...
       'layer5.csv'));
l1_idx = ismember(centroids(:,4),l_table.index);


l_table = readtable(fullfile('annotations','cortical_layers',...
       'layer2_3.csv'));
l2_idx = ismember(centroids(:,4),l_table.index);


cen1 = median(centroids(l1_idx,6));
cen2 = median(centroids(l2_idx,6));

disp(cen2/cen1)
end