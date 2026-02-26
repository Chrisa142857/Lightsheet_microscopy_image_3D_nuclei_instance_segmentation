function create_imaris_spots(centroids)
%--------------------------------------------------------------------------
% Create Imaris spot objects from centroids list. To import in Imaris, use
% the MATLAB-enabled object manager plugin
%--------------------------------------------------------------------------

scale = [1.21,1.21,4];
pos = [centroids(:,2),centroids(:,1),centroids(:,3)];
ct = centroids(:,end);
colors = [16711680,65280,255,16776960,65535,16711935];

n_cts = unique(ct);

fields = {'ColorVisible','Edges','Position','Radius','Selection','Time'};


for i = 1:length(n_cts)
   cen_idx = centroids(:,end) == n_cts(i);
   s_idx = cellfun(@(s) "vSpots_" + sprintf('%d_',i) +...
       sprintf('%d_',i) + s,fields,'UniformOutput',false);
   
   % Begin constructing structure
   s.vObjectsName(i) = {sprintf('Spots %d',i)};
   s.vObjectsType(i) = i;
   s.(s_idx{1}) = [colors(i) ,1];
   s.(s_idx{2}) = [];
   s.(s_idx{3}) = pos(cen_idx,1:3).*scale;
   s.(s_idx{4}) = ones(sum(cen_idx),3);
   s.(s_idx{5}) = [];
   s.(s_idx{6}) = zeros(sum(cen_idx),1);
end

save('ImarisObjects.mat','-struct','s')

end