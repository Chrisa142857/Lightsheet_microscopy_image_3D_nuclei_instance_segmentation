function [flatfield, darkfield]  = calculate_shading_correction(path_table, config, t_adj)
% Calculate shading correction using BaSiC
% ref: Peng T, Thorn K, Schroeder T, et al. A BaSiC tool for background and shading correction of optical microscopy images. Nat Commun. 2017;8:14836.

% Calculate darkfield image
calculate_darkfield = 'true';
sampling_freq = 0.01; 

if length(nargin) < 3
    t_adj = 1;
end

n_images = round(height(path_table)*sampling_freq);
files = round(linspace(1,height(path_table),n_images));

tempI = imread(path_table.file{1});
stack = zeros([size(tempI) length(files)], 'uint16');
for i = 1:length(files)  
    stack(:,:,i) = imread(path_table.file{files(i)});
end

stack = double(stack)*t_adj;

[flatfield,darkfield] = BaSiC(stack,'darkfield',calculate_darkfield);  

end


