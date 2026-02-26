function I = sample_stack(path_table,down_s,config)
%--------------------------------------------------------------------------
% Sample all images in stack. Images saved to output/samples if config
% structure provided.
%--------------------------------------------------------------------------
% Usage:
% I = sample_stack(path_table,spacing,config)
% 
%--------------------------------------------------------------------------
% Inputs:
% path_table: Table containing image filenames to sample.
% 
% down_s: (1x3 numeric) Amount of downsampling in [y,x,z]. 
% (default:[2,2,100])
%
% config: Configuration structure containing output directory to save image
% to.
% 
%--------------------------------------------------------------------------
% Outputs:
% I: (cell array) Downsampled 3D images for each marker in path_table.
% 
%--------------------------------------------------------------------------
if nargin<2
    down_s = [2,2,100];
end

% Get info from table
c = unique(path_table.channel_num)';
x = unique(path_table.x);
y = unique(path_table.y);
m = path_table.markers(c);

assert(length(x) == 1 && length(y) == 1, "Stack contains more than 1 tile position")
assert(isnumeric(down_s) && length(down_s) == 3, "Specify spacing as a 1x3 integer vector for downsampling")

% Get z ranges
z_min = min(path_table.z);
z_max = max(path_table.z);

z_pos = z_min:down_s(3):z_max;
if length(z_pos) == 1
    z_pos = ceil((z_max-z_min)/2);
end

% Create new image stack
tempI = read_img(path_table);
[nrows,ncols] = size(tempI);
d = round(([nrows,ncols])./down_s(1:2));
I = repmat({zeros(d(1),d(2),length(z_pos),'uint16')},1,length(c));

% Read and downsample
for i = 1:length(c)
    for j = 1:length(z_pos)
        img = read_img(path_table,[i,z_pos(j)]);
        I{i}(:,:,j) = imresize(img,[d(1),d(2)]);
    end  
end

% Write image or return
if nargout == 1
    return
else
   assert(nargin>2, "config structure required for writing images")

   save_path = fullfile(config.output_directory,'samples');   
   for i = 1:length(c)
        fname = fullfile(save_path,sprintf('%s_%s_%d_%d_%d.tif',...
            config.sample_id,m(i),down_s(1),down_s(2),down_s(3)));
        disp(fname)
        options.overwrite = true;
        options.message = true;
        saveastiff(squeeze(I{i}),char(fname),options);
   end    
end

end