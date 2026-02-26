function convert_from_imaris(config, markers, marker_order)
% Convert files from imaris

% Get image directory
if ~isequal(config.use_processed_images,"false")
    img_directory = fullfile(config.output_directory, config.use_processed_images);
else
    img_directory = config.img_directory;
end

files = dir(fullfile(img_directory,'*.tif'));
filenames = {files.name};

% Seperate files into parts
parts = cellfun(@(s) strsplit(s, {'_','.'}), filenames, 'UniformOutput', false);


for i = 1:length(files)
    sample = parts{i}{1};
    z = str2double(parts{i}{4}(2:end))+1;
    channel_num = str2double(parts{i}{3}(2:end))+1;
    marker = markers(channel_num);
    channel_num = marker_order(channel_num);
    
    newname = sample + "_" + sprintf('%04d_C%d_%s_stitched.tif',z,channel_num,marker);
    
    cmd = sprintf("mv %s %s", fullfile(img_directory, filenames{i}),...
        fullfile(img_directory,newname));
    
    system(cmd);

end


end