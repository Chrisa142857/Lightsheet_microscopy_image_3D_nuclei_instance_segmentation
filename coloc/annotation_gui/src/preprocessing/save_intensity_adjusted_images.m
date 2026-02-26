function save_intensity_adjusted_images(config)
%--------------------------------------------------------------------------
% Save adjusted images to output directory. Used as a peripheral function
% and not part of the typical workflow.
%--------------------------------------------------------------------------

% Load path table
path_table = path_to_table(config, 'raw');

% Load adjustment parameters
adj_params = load_var(config,'adj_params');

% Create output directory
save_path = fullfile(config.output_directory, 'intensity_adjusted');
if ~isfolder(save_path)
    mkdir(save_path)
end

% Read image, apply adjustment, and write
for i = 1:height(path_table)
    img = read_img(path_table(i,:).file);
    marker = path_table(i,:).markers;
    y = path_table(i,:).y;
    x = path_table(i,:).x;
    z = path_table(i,:).z;
    c = path_table(i,:).channel_num;
    
    img_adjusted = apply_intensity_adjustment(img, adj_params.(marker),...
        'r',y,'c',x);
    
    fname = config.sample_id + sprintf("_%.4d_C%d_%s_%.2d_%.2d_adjusted.tif",...
        z, c, marker, y, x);
    
    imwrite(img_adjusted, fullfile(save_path,fname))
end

end