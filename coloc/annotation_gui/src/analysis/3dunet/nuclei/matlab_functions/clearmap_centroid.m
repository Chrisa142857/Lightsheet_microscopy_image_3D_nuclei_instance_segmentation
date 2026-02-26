%% Get centroids from CUBIC results
function centroids = clearmap_centroid(idx,resolution)

results_path = fullfile(pwd,'Updated Training Samples', 'ClearMap',...
    'results',resolution);

files = dir(results_path);

array = readmatrix(fullfile(results_path,files(idx).name));

x = array(:,1)+1;
y = array(:,2)+1;
z = array(:,3)+1;

centroids = [x y z];

end

