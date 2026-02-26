%% Get centroids from CUBIC results
function centroids = cubic_centroid(idx,resolution)

results_path = fullfile(pwd,'Updated Training Samples', 'CUBIC',...
    'results',resolution);

files = dir(results_path);

array = readmatrix(files(idx).name);

x = array(:,2)+1;
y = array(:,3)+1;
z = array(:,4)+1;

centroids = [y x z];

end

