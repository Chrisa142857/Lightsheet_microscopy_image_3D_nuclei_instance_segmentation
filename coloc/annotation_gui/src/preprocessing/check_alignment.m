function img = check_alignment(config, ranges, spacing, use_table)
%--------------------------------------------------------------------------
% Check image alignment of channels and save to samples directory.
%--------------------------------------------------------------------------
% Usage:
% img = check_alignment(config, ranges, spacing, use_table)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Configuration structure from NM_process.
%
% ranges: (1x2 or 1x3 cell array) Tile row and column. 3rd value can be a 
% range of slices in the slices (e.g. 1:100). Accepts only single tile 
% position.
%
% spacing: (1x3 integer) Amount of downsampling for each dimension in each
% stack. (default: [3,3,20])
%
% use_table: (logical) Check translations from alignment table. Otherwise,
% will use images saved directly in aligned directory. (default: false)
%
%--------------------------------------------------------------------------
% Outputs:
% img: 4D aligned image stacks.
% 
%--------------------------------------------------------------------------
if ~iscell(ranges)
    error("Second input should be 1x2 cell array containing tile row, column position. "+...
        "Optional 3rd cell index to subset a range of tile slices. \n")
end

if nargin<3 || isempty(spacing)
    spacing = [3,3,20];
end

if nargin<4
    use_table = false;
end

if isfield(config,'sample_id')
    sample_id = string(config.sample_id);
else
    sample_id = "SAMPLE1";
end

fprintf("Loading image information \n")

% Create sample alignment directory if not present
save_directory = fullfile(config.output_directory,'samples','alignment');
if ~isfolder(save_directory)
    mkdir(save_directory);
end

% Load images from aligned directory
align_folder = fullfile(config.output_directory,'aligned');
pre_align = false;
if ~use_table && isfolder(align_folder)
    config.img_directory = fullfile(config.output_directory,'aligned');
    try
        path_table = path_to_table(config,'aligned',true,false);
    catch
        path_table = path_to_table(config,'aligned',false,false);
    end
else 
    % Load from alignment table
    table_path = fullfile(config.output_directory,'variables','alignment_table.mat');
    if isfile(table_path)
        load(table_path,'alignment_table')
        fprintf("Loading transforms from alignment table \n") 
        path_table = path_to_table(config,'raw');
        pre_align = true;
    else
        warning("Could not locate aligned images or alignment table variable")
        return
    end
end

% Subset markers,x,y positions
path_table = path_table(path_table.y == ranges{1} & ...
    path_table.x == ranges{2},:);
if pre_align
    alignment_table = alignment_table{ranges{1},ranges{2}};
    vars = alignment_table.Properties.VariableNames;
    f_idx = contains(vars,'file');
end
if isempty(path_table)
    warning("Tile position specified could not be found")
    return
end

% Check markers present
markers = unique([path_table.markers ,path_table.channel_num],'rows');
[~,idx] = sort(markers(:,2)); % sort just the first column
markers = markers(idx,1);

% Get z ranges
if length(ranges) == 2
    z_min = min(path_table.z);
    z_max = max(path_table.z);
elseif length(ranges{3}) == 1
    z_min = 1;
    z_max = max(ranges{3});
else
    z_min = min(ranges{3});
    z_max = max(ranges{3});
end

% Subset z positions
z_pos = z_min:spacing(3):z_max;
path_table = path_table(ismember(path_table.z,z_pos),:);
if pre_align
    alignment_table = alignment_table(ismember(alignment_table.Reference_Z,z_pos),:);
    align_files = alignment_table{:,f_idx};
    c_idx = repmat({0},1,length(markers));
    for i = 1:length(c_idx)
       c = find(contains(vars,'X') & contains(vars,markers(i)));
       if ~isempty(c)
           c_idx{i} = c;
       end
    end
end

% Get size of resampled images
tempI = imread(path_table.file{1});
[nrows,ncols] = size(tempI);
nrows_d = round(nrows/spacing(1));
ncols_d = round(ncols/spacing(2));

% Create composite image
img = zeros(nrows_d,ncols_d,length(z_pos),length(markers),'uint16');

for i = 1:length(markers)
    fprintf("Reading marker %s \n",markers(i)) 
    path_sub = path_table(path_table.markers == markers(i),:);
    
    for j = 1:length(z_pos)       
       % Pre-align from alignment table
       if ~pre_align    
           I = imread(path_sub.file{j});
       else
          if ~isempty(align_files{j,i})
              I = imread(align_files{j,i});
          else
              continue
          end
          
          if ~isempty(c_idx{i}) && c_idx{i}>0
            % Apply translation
            x = alignment_table{j,c_idx{i}};
            y = alignment_table{j,c_idx{i}+1};
            I = imtranslate(I,[x y]);
          end
       end
       img(:,:,j,i) = imresize(I,[nrows_d,ncols_d]);
    end
    
    if nargout <1
        fname = fullfile(save_directory,sprintf('%s_%s_%d_%d.tif',...
            sample_id,markers(i),ranges{1},ranges{2}));
        options.overwrite = true;
        options.message = false;
        saveastiff(squeeze(img(:,:,:,i)),char(fname),options);
    end
end

if nargout<1; clear img; end

end