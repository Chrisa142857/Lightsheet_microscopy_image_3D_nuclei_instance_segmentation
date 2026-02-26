function out = elastix_channel_alignment(config, path_table, warp_images)
%--------------------------------------------------------------------------
% Aligning multiple channels to a reference nuclei channel using a coarse
% non-linear B-Spline registration. This uses a modified version of
% melastix, a MATLAB wrapper for the elastix registration toolbox. Melastix
% will need to be in your MATLAB path and Elastix libraries will need to be
% properly setup. In its current implementation, this function contains many
% parameters that were empirically adjusted to account for subtle
% deformations. Further fine-tuning is likley necessary if the expected
% deformations are large. Also note, this has relatively high memory
% requirements if the image stacks are large. Outputs are 2D series of
% aligned channel images + reference channel images in the 'aligned'
% directory
%--------------------------------------------------------------------------

% Unpack image information variables
sample_id = config.sample_id;
markers = config.markers;
output_directory = config.output_directory;
home_path = config.home_path;
align_channels = config.align_channels;

% Parameters not found in NMp_template. Defaults
img_gamma_adj = repmat(0.5,1,length(markers)); % Apply gamma during intensity adjustment. Decrease for channels with low background
smooth_blobs = "false"; % Smooth blobs using gaussian filter
apply_dog = "true"; % Apply difference of gaussian

% Registration-specific parameters
elastix_params = config.elastix_params;
max_chunk_size = config.max_chunk_size; 
chunk_pad = config.chunk_pad;
mask_int_threshold = config.mask_int_threshold;
s = config.resample_s;
histogram_bins = config.hist_match;
if length(histogram_bins) == 1
    histogram_bins = repmat(histogram_bins,1,length(markers)-1);
end

% Unpack intensity adjustment variables
lowerThresh = config.lowerThresh;
upperThresh = config.upperThresh;
signalThresh = config.signalThresh;

% Check if equal resolution
equal_res = cellfun(@(s) all(config.resolution{1}(1:2) == s(1:2)),config.resolution);
if ~all(equal_res)
    res_adj = cellfun(@(s) config.resolution{1}(1:2)./s(1:2),config.resolution,'UniformOutput',false);
else
    res_adj = repmat({ones(1,2)},1,length(markers));
end
    
% Get x,y tile positions from path_table
x = unique(path_table.x);
y = unique(path_table.y);
assert(length(x) == 1, "More than 1 tile column detected in path_table.")
assert(length(y) == 1, "More than 1 tile row detected in path_table.")

% Paths to elastix parameter files
parameter_path = fullfile(home_path,'data','elastix_parameter_files','channel_alignment');
transform_path = repmat({''},length(markers)-1,3); outputDir = cell(1,length(markers)-1);

% Set defaults parameters if empty
if isempty(elastix_params)
    elastix_params = repmat("",1,length(markers)-1);
    for i = 1:length(markers)-1
        if lowerThresh(i+1)/upperThresh(i+1) <=0.1 && isequal(config.pre_align,"true")
            % 32 bins prealign
            elastix_params(i) = "32_prealign";
            max_chunk_size = length(unique(path_table.z))+100;
            chunk_pad = 10;
        elseif lowerThresh(i+1)/upperThresh(i+1) >0.1 && isequal(config.pre_align,"true")
            % 16 bins prealign
            elastix_params(i) = "16_prealign";
            max_chunk_size = length(unique(path_table.z))+100;
            chunk_pad = 10;
        elseif lowerThresh(i+1)/upperThresh(i+1) <=0.1 && isequal(config.pre_align,"false")
            % 32 bins
            elastix_params(i) = "32_bins";
            max_chunk_size = 300;
            chunk_pad = 30;
        else
            % 16 bins
            elastix_params(i) = "16_bins";
            max_chunk_size = 300;
            chunk_pad = 30;
        end
    end
end

% Create cell array containing parameters
for i = 1:length(markers)-1
    parameter_dir = dir(parameter_path);
    idx = find({parameter_dir.name} == elastix_params(i), 1);
    if ~isempty(idx)
        % Detect which transform is in the parameter file
        parameterSub = dir(fullfile(parameter_path,elastix_params(i)));
        parameterSub = parameterSub(arrayfun(@(s) endsWith(s.name,'.txt'),parameterSub));
        for j = 1:length(parameterSub)
            file_path = fullfile(parameterSub(1).folder,parameterSub(j).name);
            text = textread(file_path,'%s','delimiter','\n');
            n = find(cellfun(@(s) contains(s,'(Transform '),text));
            if contains(text(n),'Translation')
                transform_path{i,1} = file_path;
            elseif contains(text(n),'Affine') || contains(text(n),'Euler')
                transform_path{i,2} = file_path;
            elseif contains(text(n),'BSpline')
                transform_path{i,3} = file_path;
            end
        end
        if ismember(i+1,align_channels)
            fprintf('Using parameters %s for marker %s \n', transform_path{i,3}, config.markers(i+1))
        end
    else
        error("Could not locate elastix parameter folder %s "+...
            "in %s",elastix_params(i),parameter_path)
    end
    % Create empty tmp directory for saving images/transforms
    outputDir{i} = string(fullfile(config.output_directory,sprintf('tmp%d',i)));
    if ~exist(outputDir{i},'dir')
        mkdir(outputDir{i})
    else
        rmdir(outputDir{i},'s')
        mkdir(outputDir{i})
    end
end

fprintf('Using max chunk size of %d with %d slice padding \n', max_chunk_size, chunk_pad)

%% Check if previous registration parameters exist
varfile = fullfile(output_directory,'variables','alignment_params.mat');   
using_loaded_parameters = false;
out = [];
if isfile(varfile) && ~isequal(config.pre_align,"true")
    m = matfile(varfile);
    out = m.alignment_params(y,x);
    if ~isempty(out{1})
        fprintf("%s\t Loading previous alignment parameters \n",datetime('now'))
        out = out{1};
        using_loaded_parameters = true;
    else
        out = [];
    end
end

% Load pre-alignment table if exists
if isequal(config.pre_align,"true") && isfield(config,'alignment_table')
    alignment_table = config.alignment_table;
else
    alignment_table = [];
end

%% Subset channels, chunks, and/or ranges
% Subset channels or use all
if ~isempty(config.align_channels)
    align_channels = config.align_channels;
    align_channels(align_channels == 1) = [];
    assert(all(ismember(align_channels,2:length(markers))), "Channels to align "+...
        "out of range in image dataset.")
    c_idx = [1,ismember(2:length(markers),align_channels)];
else
    align_channels = 2:length(markers);
    c_idx = ones(1,length(markers));
end
align_markers = markers(align_channels);
nslices = height(path_table(path_table.markers == markers(1),:));

% Check whether aligning full stack
update_stack = false;
if isequal(config.channel_alignment,"update")
    update_stack = true;
    assert(~(~isempty(config.align_chunks) && ~isempty(config.align_slices)),....
        "Can update either chunks or slices but not both.")
    if ~isempty(config.align_slices) || ~isempty(config.align_chunks)
        assert(~isempty(out),...
            "Cannot update chunks or slices without running alignment on full tile stack first.")
        assert(out.nslices == nslices,...
            "Image numbers do not match between loaded parameters and file path table.") 
        assert(all(isfield(out,align_markers)),...
            "Not all markers to align are present in loaded parameters")        
    end
end

% Subset z positions or use all
if update_stack && ~isempty(config.align_slices)
    if ~iscell(config.align_slices)
        config.align_slices = {config.align_slices};
    end
    
    % Subset by slice range   
    chunk_ranges = zeros(length(config.align_slices),4);
    for i = 1:length(config.align_slices)
        sl = config.align_slices{i};
        chunk_ranges(i,:) = [min(sl), max(sl), max(1,min(sl)-chunk_pad), min(max(sl)+chunk_pad,nslices)];
        fprintf('Aligning slices %d to %d\n', chunk_ranges(i,1), chunk_ranges(i,2))
    end
    
    % Set n chunks to 1 and specify start and end positions
    align_chunks = 1:size(chunk_ranges,1);
    z_min_adj = min(chunk_ranges(:,3));
    z_max_adj = max(chunk_ranges(:,4));
    z_range_save = unique([config.align_slices{:}]);
    
elseif update_stack && ~isempty(config.align_chunks)
    % Subset by chunk
    align_chunks = config.align_chunks;
    
    % Get ranges for each chunk for each marker to be aligned
    chunk_ranges = cell(1,length(markers));
    for i = 1:length(align_channels)
        ranges = cellfun(@(s) s.chunk_ranges, out.(align_markers(i)).chunk_tform,...
            'UniformOutput', false);        
        chunk_ranges{align_channels(i)} = cat(1,ranges{align_chunks});
    end
    
    % Rule: if aligning multiple channels, z ranges should be consistent
    % between channels. If not, must align each channel individually
    if length(align_channels) > 1
        chk = cellfun(@(s) all(s == chunk_ranges{2},'all'),chunk_ranges(2:end));
        assert(all(chk), "If updating chunks in multiple channels, all chunk "+...
            "start/end positions must be consistent. Otherwise, update "+...
            "chunk alignment 1 channel at a time.")
    end
    chunk_ranges = chunk_ranges{2};
    fprintf('Aligning chunks %d\n', align_chunks)
    
    % Stack all ranges to specify which images to read
    z_min_adj = min(chunk_ranges(:,3));
    z_max_adj = max(chunk_ranges(:,4));
    
    % Specify which slices to save after transforming
    slices = arrayfun(@(s,t) s:t, chunk_ranges(:,1), chunk_ranges(:,2), 'UniformOutput', false);
    z_range_save = unique([slices{:}]);
    
else
    % Use all images
    z_min_adj = 1;
    z_max_adj = nslices;
    align_chunks = NaN;
    z_range_save = 1:nslices;
    if update_stack && ~isempty(out)
        for i = 1:length(align_markers)
            if isfield(out,align_markers(i))
                out = rmfield(out,align_markers(i));
            end
        end
    end
end

%% Preconfigure matrixes
tic
% Get basic image and file information 
z_range_adj = z_min_adj:z_max_adj;
tempI = imread(path_table.file{1});
[nrows, ncols] = size(tempI);
cropping_flag =  [0 0 0]; % Flag for cropping images to reference channel
dim_adj = round([nrows ncols nslices]./s); % Resampled image dimensions

% Check if channels have the same sized images
for i = [1,align_channels]
    path_sub = path_table(path_table.markers == markers(i),:);
    tempI2 = imread(path_sub.file{1});    
    if size(tempI) ~= size(tempI2)
        cropping_flag(i) = 1;
    end
end

% Initialize matrices for storing raw images
I_raw = cell(1,length(markers));
I_raw(:) = {zeros(nrows,ncols,nslices,'uint16')};

% Start parallel pool
try
    p = gcp('nocreate');
catch
    p = 1;
end

if isempty(p) && length(config.align_channels)>1
    parpool
    p = gcp('nocreate');
    max_workers = p.NumWorkers;
else
    max_workers = 0;
end

%% Read the images
for i = 1:length(markers)
    % Read image slices for only the channels that will be aligned
    if i > 1 && c_idx(i) == 0
        continue
    end
    fprintf('Reading images and pre-processing marker %s \n', markers(i))
    
    % If using pre-aligned, account for a shift in z by reading images from
    % alignment table and noting shifts   
    if ~update_stack && using_loaded_parameters &&...
            isfield(out,markers(i)) && isfield(out.(markers(i)),'pre_alignment')
        % Load pre-alignment from alignment parameters
        pre_align_params.(markers(i)) = out.(markers(i)).pre_alignment;
        path_sub = pre_align_params.(markers(i)).file;
        
        % Check that files exist
        if ~all(cellfun(@(s) isfile(s) | isempty(s), path_sub))
            error("Files in loaded alignment table do not exist")
        end

    elseif ~isempty(alignment_table) && i>1
        % Load from alignment table
        path_sub = path_table(path_table.markers == markers(1),:).file;
        idx = arrayfun(@(s) find(string(s{1}) == string(alignment_table{:,1})),...
            path_sub,'UniformOutput',false);
        for j = 1:length(idx)
            if isempty(idx{j})
                pre_align_params.(markers(i)).file(j) = {''};
                pre_align_params.(markers(i)).x(j) = NaN;
                pre_align_params.(markers(i)).y(j) = NaN;
            else
                pre_align_params.(markers(i)).file(j) = alignment_table{idx{j},"file_" + num2str(i)};
                pre_align_params.(markers(i)).x(j) = alignment_table{idx{j},"X_Shift_" + markers(i)};
                pre_align_params.(markers(i)).y(j) = alignment_table{idx{j},"Y_Shift_" + markers(i)};
            end
        end
        path_sub = pre_align_params.(markers(i)).file;
        
    else
        % No prealignment
        path_sub = path_table(path_table.markers == markers(i),:).file;
    end
    
    % Read images
    I_sub = zeros(nrows,ncols,length(z_range_adj));
    files = path_sub(z_range_adj);
    res_adj1 = res_adj{i}; cropping_flag1 = cropping_flag(i);
    for j = 1:length(z_range_adj)        
        if ~isempty(files{j})
            I_sub(:,:,j) = read_slice(files{j},res_adj1,cropping_flag1);
        else
            I_sub(:,:,j) = zeros(nrows,ncols,'uint16');
        end
    end
    I_raw{i}(:,:,z_range_adj) = I_sub;
end

% Apply intensity adjustments for each tile
if any(arrayfun(@(s) isequal(s,"true"),config.adjust_intensity)) &&...
   all(arrayfun(@(s) ~isequal(s,"false"),config.adjust_tile_shading))
   adj_params = arrayfun(@(s) config.adj_params.(s),markers,'UniformOutput',false);
   adj_ref = adj_params{1};
    parfor (i = 1:length(markers),max_workers)
        if c_idx(i) == 0
            continue
        else
            fprintf('Applying intensity adjustments for marker %s\n',markers(i));
        end
        I_raw{i} = apply_intensity_adjustments_tile(I_raw{i}, adj_params{i}, adj_ref, z_range_adj);
    end
end

% Apply pre-alignments if necessary
if exist('pre_align_params','var') == 1
    for i = align_channels
        fprintf('Pre-aligning marker %s \n', markers(i))
        x_shift = pre_align_params.(markers(i)).x;
        y_shift = pre_align_params.(markers(i)).y;
        
        for j = 1:length(z_range_adj)
            if ~isnan(x_shift(j)) && ~isnan(y_shift(j))
                % Apply translation
                I_raw{i}(:,:,z_range_adj(j)) = imtranslate(I_raw{i}(:,:,z_range_adj(j)),...
                    [x_shift(z_range_adj(j)) y_shift(z_range_adj(j))]);
            end
        end
    end
end

% Further adjustments to optimize images for registration
if ~using_loaded_parameters || update_stack
    % Generate mask
    if update_stack && ~isempty(config.align_slices)
        % If specific slices selected, use all the features in the slice
        % objects
        mask = zeros(size(I_raw{1}));
    else
        % Generate sampling mask from intensity threshold
        mask = generate_sampling_mask(I_raw{1},mask_int_threshold,signalThresh(1));
    end
    
    % Prepare images for registration
    I = cell(1,length(I_raw));
    for i = [1, align_channels]
        I{i} = I_raw{i};
        
        % Smooth blobs using Guassian filter to reduce effects of
        % noise. Not a big impact if resampling
        if isequal(smooth_blobs,"true")
            for j = 1:length(z_range_adj)
                I{i}(:,:,z_range_adj(j)) = imgaussfilt(I{i}(:,:,z_range_adj(j)),1);
            end
        end
        
        % Set the mask to zero at border regions if pre-aligned
        if ~isempty(alignment_table) && i>1
            mask(I{i} == 0) = 0;                    
        end
        
        % Set the mask to zero at bright regions if aligning slices
        if update_stack && ~isempty(config.align_slices) && i==1
           upperThresh1 = upperThresh(1)*65535*0.9;
           mask(I{1}>upperThresh1) = 0;
        end
        
        % Adjust intensity 
        I{i} = imadjustn(I{i},[signalThresh(i) upperThresh(i)],[],img_gamma_adj(i));
        
        % Smooth blobs using Guassian filter to reduce effects of
        % noise. Not a big impact if resampling
        if isequal(apply_dog,"true")
            for j = 1:length(z_range_adj)
                I{i}(:,:,z_range_adj(j)) = dog_adjust(I{i}(:,:,z_range_adj(j)),...
                    config.nuc_radius);
            end
        end
        
        % Resample image
        I{i} = im2int16(imresize3(I{i},dim_adj));
    
        % Histogram matching
        if i>1 && ~isempty(histogram_bins) && histogram_bins(i-1) > 0
            I{i}(:,:,z_range_adj) = imhistmatch(I{i}(:,:,z_range_adj),...
                I{1}(:,:,z_range_adj),histogram_bins(i-1));
        end 
    end
    
    % Resize the mask 
    mask = imresize3(single(mask),dim_adj,'nearest');
    if update_stack && ~isempty(config.align_slices)
        mask(:,:,z_range_save) = 1;
        fprintf('Using masked ROI for slices \n')
    end

    % Determine chunk positions
    if isempty(config.align_chunks)
        % If total number of images is less than the max chunk size, adjust
        % padding
        if max_chunk_size>nslices
           chunk_pad = 0; 
        end
        chunk_ranges = get_chunk_positions(mask, nslices, max_chunk_size, chunk_pad);
        align_chunks = 1:size(chunk_ranges,1);
    end
end
fprintf('Images loaded and pre-processed in %d seconds\n', round(toc))

%% Perform the registration
if ~using_loaded_parameters || update_stack
    % First save any prealignment parameters
    if exist('pre_align_params','var')
        pre_fields = fields(pre_align_params);
        for i = 1:length(pre_fields)
            out.(pre_fields{i}).pre_alignment = pre_align_params.(pre_fields{i});
        end
    end
    
    % Next perform initial registration by translation on whole downsampled stack
    fprintf('Performing intial registration \n')
    
    % If doing slices or chunks, check for init_tform 
    init_tform = cell(1,length(markers)-1);
    if update_stack && using_loaded_parameters &&...
            ~isempty(config.align_slices) && isempty(alignment_table)
        % Load initial tform it exists
        for i = 1:length(align_markers)
            if isfield(out.(align_markers(i)),'init_tform')
                fprintf("Loading intial whole stack registration for marker %s\n", align_markers(i))
                init_tform{align_channels(i)-1} = out.(align_markers(i)).init_tform;
                transform_path{align_channels(i)-1,1} = '';
            end
        end
    else
        I1 = I{1}(:,:,z_range_adj);
        parfor (i = 1:length(markers)-1,max_workers)
           if c_idx(i+1) == 0 || isempty(transform_path{i,1})
                if c_idx(i+1) == 1
                    fprintf("Skipping intial whole stack registration for marker %s\n", markers(i+1))
                end
                continue
            elseif i ~= 1
                pause(1)
           end
           
           % Perform registration
           init_tform{i} = elastix(I{i+1}(:,:,z_range_adj),I1,outputDir{i},...
               transform_path(i,1),'s',s,'threads',[]);

            % Parameters are as -x,-y,-z
            fprintf("Intial transform parameters: %s\n",sprintf("%.3f\t",...
                init_tform{i}.TransformParameters{1}.TransformParameters))
        end
        clear I1
    end
        
    % Save inital transform 
    for i = 1:length(markers)-1
        if ~isempty(init_tform{i})
            out.(markers(i+1)).init_tform = init_tform{i};
        end
    end
    
    % Save tile positions
    out.x = x; out.y = y;
    out.ncols = ncols; out.nrows = nrows; out.nslices = nslices;
    
    fprintf('Performing chunk-wise registration \n')
    for i = 1:length(align_chunks)
        fprintf('Aligning chunk %d out of %d\n', i, length(align_chunks))
        
        % Get loaded chunk start and end location
        chunk_start = chunk_ranges(i,3); chunk_end = chunk_ranges(i,4);

        % Take the mask for the respective chunk to align
        mask_chunk = mask(:,:,chunk_start:chunk_end);
        
        % Clip the bottom and top sections in padded regions
        if chunk_start - chunk_ranges(i,1) > 1
            mask_chunk(:,:,chunk_start - chunk_ranges(i,1)) = 0;
        end        
        if chunk_end - chunk_ranges(i,2) > 1
            a = chunk_end - chunk_ranges(i,2);
            mask_chunk(:,:,size(mask_chunk,3)-a:size(mask_chunk,3)) = 0;
        end
        
        % Take chunk images from reference channel
        I_ref = I{1}(:,:,chunk_start:chunk_end);

        % Create cell array for storing chunk registration parameters
        chunk_out = cell(1,length(markers)-1);
        
        tic
        parfor (j = 1:length(markers)-1,max_workers)
           if c_idx(j+1) == 0
                continue
            elseif j ~= 1
                pause(1)
           end
           
            % Remove empty transform
            transforms = transform_path(j,:);
            transforms_sub = transforms(cellfun(@(s) ~isempty(s),transforms));
            if length(transforms_sub) > 2
                transforms_sub = transforms_sub(2:end);
            elseif isempty(transforms_sub)
                continue
            end
                        
            if ~isempty(init_tform{j})
                % Take initial transform of whole stack and save it as an
                % elastix parameters file
                init_tform{j}.TransformParameters{1}.Size(3) = size(mask_chunk,3);
                init_tform_path = sprintf('%s/init_transform_C%d_%d_%d_%d_%s.txt',outputDir{j},j,y,x,i,sample_id);
                elastix_paramStruct2txt(init_tform_path,init_tform{j}.TransformParameters{1})

                % Perform registration on the chunk for this channel
                chunk_out{j} = elastix(I{j+1}(:,:,chunk_start:chunk_end),I_ref,...
                    outputDir{j},transforms_sub,'fMask', mask_chunk,...
                    't0',init_tform_path,'s',s,'threads',[]);

                % Save initial transform filename into structure
                chunk_out{j}.TransformParameters = horzcat(init_tform{j}.TransformParameters,...
                    chunk_out{j}.TransformParameters);
            else
                % Perform registration on the chunk for this channel
                chunk_out{j} = elastix(I{j+1}(:,:,chunk_start:chunk_end),I_ref,...
                    outputDir{j},transforms_sub,'fMask', mask_chunk,'s',s,'threads',[]);
            end
            
            % Reset temporary directory
            rmdir(outputDir{j},'s')
            mkdir(outputDir{j})
        end

        % Save registration parameters into structure
        for j = 1:length(markers)-1
           if c_idx(j+1) == 0
                continue
           end
            chunk_out{j}.chunk_ranges = chunk_ranges(i,:);
            out.(markers(j+1)).chunk_tform{i} = chunk_out{j};
        end
        
        fprintf('Finished registration in %d seconds\n', round(toc))
    end
    
    % Cleanup temporary directories
    cellfun(@(s) rmdir(s,'s'), outputDir)
end

% Return if not aplying transformations
if ~warp_images || isequal(config.save_images,'false')
    return
elseif using_loaded_parameters && ~update_stack
    fprintf("%s\t Transform parameters already exist. Skipping registration "+... 
        "and applying transforms.\n",datetime('now'));
end

%% Apply transformations to adjusted images
% Subset chunks that we're interested in aligning
for i = 1:length(align_markers)
    if isfield(out,align_markers(i))
        out_sub = out.(align_markers(i));
        if isfield(out_sub,'chunk_tform')
            if isnan(align_chunks)
                align_chunks = 1:length(out_sub.chunk_tform);
            end
            out2.chunk_tform = out_sub.chunk_tform(align_chunks);
        end
        if isfield(out_sub,'init_tform')
            out2.init_tform = out_sub.init_tform;
        end
    end
    I_raw{align_channels(i)} = apply_transformations(I_raw{align_channels(i)},out2,max_workers);
end

%% Save the images
if isequal(config.save_images,"true")
    % Create directory to store images
    if exist(fullfile(output_directory,'aligned'),'dir') ~= 7
        mkdir(fullfile(output_directory,'aligned'));
    end

    if update_stack && ~isempty(config.align_slices)
        z_range_save = unique([config.align_slices{:}]);
        fprintf("%s\t Saving only slices between %d and %d \n",datetime('now'),min(z_range_save),max(z_range_save))
    end

    % Save aligned images
    for i = 1:length(markers)
       if i > 1 && c_idx(i) == 0
            continue
       end
        markers1 = markers(i);
        I = I_raw{i}(:,:,z_range_save);
        fprintf("%s\t Writing aligned images for marker %s\n",datetime('now'),markers1)
        parfor (j = 1:length(z_range_save),max_workers)
            img_name = sprintf('%s_%s_C%d_%s_0%d_0%d_aligned.tif',...
                sample_id,num2str(z_range_save(j),'%04.f'),i,markers1,y,x);
            img_path = fullfile(output_directory,'aligned',img_name);        
            %fprintf("%s\t Writing aligned image %s\n",datetime('now'),img_name)
            imwrite(I(:,:,j),img_path)
        end
    end
end

end


function img = read_slice(file, res_adj, cropping_flag)
% Read image and apply adjustments
img = imread(file);

% Resample image to image reference image if different
% resolutions
if isequal(res_adj,ones(1,2))
    img = imresize(img,round(size(img)./res_adj),'Method','bicubic');
end

% Crop or pad image if it is not the same size as reference image
if cropping_flag == 1
    img= crop_to_ref(tempI,img);
end

end


function I_raw = apply_transformations(I_raw,out,max_workers)
% Apply elastix transform parameters
% I_raw should just be 3D matrix for 1 channel
% out should contain fields for chunk transforms and initial transforms

[nrows, ncols, nslices] = size(I_raw);
nchunks = length(out.chunk_tform);

% Get chunk sizes and locations in the stack with/without padding
chunk_start = cellfun(@(s) s.chunk_ranges(1),out.chunk_tform);
chunk_end = cellfun(@(s) s.chunk_ranges(2),out.chunk_tform);
chunk_start_adj = cellfun(@(s) s.chunk_ranges(3),out.chunk_tform);
chunk_end_adj = cellfun(@(s) s.chunk_ranges(4),out.chunk_tform);

top_trim = zeros(1,nchunks);
bottom_trim = zeros(1,nchunks);

% Create cell array containing chunks
I_chunk = cell(1,nchunks);
chunk_tforms = cell(1,nchunks);
for i = 1:nchunks
    if i == nchunks && chunk_end(i) == nslices
        a = 0;
    else
        a = 1;
    end
    
   % Get locations for current chunk
   I_chunk{i} = im2int16(I_raw(:,:,chunk_start_adj(i):chunk_end_adj(i)));

   % Get regions to trim and make adjustment at end position
   top_trim(i) = abs(chunk_start_adj(i)-chunk_start(i));
   bottom_trim(i) = abs(chunk_end_adj(i)-chunk_end(i))+a;
   chunk_end(i) = chunk_end(i)-a;
      
   % Adjust size back from rescaled resolution
   chunk_tforms{i} = out.chunk_tform{i};
   size1 = chunk_tforms{i}.TransformParameters{end}.Size;
   size1 = [ncols nrows size1(3)];
   
   % Add intial transform if present and number of chunk transforms <3
   if length(chunk_tforms{i}.TransformParameters)<3 && isfield(out,'init_tform')
       chunk_tforms{i}.TransformParameters = horzcat(out.init_tform.TransformParameters,...
           chunk_tforms{i}.TransformParameters);
   end

    % Note using int32. Weird pixel errors if using uint/double data
    % types. Adjust sizes and adjust default settings.
    for j = 1:length(chunk_tforms{i}.TransformParameters)
        chunk_tforms{i}.TransformParameters{j}.Size = size1;
        chunk_tforms{i}.TransformParameters{j}.Spacing = [1 1 1];
        chunk_tforms{i}.TransformParameters{j}.DefaultPixelValue = -32768;
        chunk_tforms{i}.TransformParameters{j}.InitialTransformParametersFileName = 'NoInitialTransform';
    end
end

% Apply transformatin for each chunk
I_trans = cell(1,nchunks);
tic
parfor (i = 1:nchunks,max_workers)
    % Apply transform
    I_trans{i} = transformix(I_chunk{i},chunk_tforms{i},[1 1 1],[]);

    % Trim ends
    z_range = 1+top_trim(i):size(I_trans{i},3)-bottom_trim(i);
    I_trans{i} = im2uint16(int16(I_trans{i}(:,:,z_range)));
end

% Replace in image stack
for i = 1:nchunks
    I_raw(:,:,(chunk_start(i):chunk_end(i))) = I_trans{i};   
end

fprintf('Finished transformation in %d seconds\n', round(toc))

end


function mask = generate_sampling_mask(I,mask_int_threshold,signalThresh)
%Generate mask
slice_thresh = 0.01;

% Downsample to 10% resolution
I2 = imresize3(I,0.10,'linear'); 

% Set mask intensity threshold if empty
if isempty(mask_int_threshold)
    mask_int_threshold = signalThresh;
end

signal = 0;
while signal < 0.05
    % Binarize mask and fill holes
    mask = imbinarize(I2,mask_int_threshold);
    mask = imfill(mask,26,'holes');

    % Clear any slices with only a few 
    idx = sum(sum(mask,1),2)/numel(mask(:,:,1))<slice_thresh;
    mask(:,:,idx) = 0;

    % Keep only brightest compnent. Disconnected components will give errors
    %labels = bwconncomp(mask);
    %intensity = regionprops3(labels,I2,{'VoxelIdxList','MeanIntensity',});
    %idx = find(intensity.MeanIntensity == max(intensity.MeanIntensity));
    %for i = 1:height(intensity)
    %    if i ~= idx
    %        mask(intensity.VoxelIdxList{i}) = 0;
    %    end
    %end

    % Resize back to original size
    mask = imresize3(double(mask), size(I),'method', 'nearest');
    signal = sum(mask(:))/numel(mask);
    mask_int_threshold = mask_int_threshold*0.95;
end
fprintf('\n Using a mask intensity threshold of %.4f \n', mask_int_threshold)
fprintf('\n Using %.4f percent of voxels \n', signal*100)

end


function chunk_ranges = get_chunk_positions(mask, total_images, max_chunk_size, chunk_pad)
% Get chunk position from binary mask

ind = find(sum(sum(mask,1),2) > 0);
z_start = min(ind);
z_end = max(ind);
z_pos = z_start:z_end;

nchunks = 1;
chunk_size = length(z_pos);

if chunk_size < max_chunk_size
    chunk_start = max(z_start - chunk_pad,1);
    chunk_end = min(z_end + chunk_pad,total_images);

    chunk_start_adj = chunk_start;
    chunk_end_adj = chunk_end;
else
    while chunk_size > max_chunk_size
        nchunks = nchunks + 1;
        chunk_size = ceil(length(z_pos)/nchunks);
    end
    chunk_start = z_start:chunk_size:z_end;
    chunk_end = chunk_start + chunk_size;

    chunk_start(1) = z_start;
    chunk_end(end) = z_end;

    chunk_start_adj = chunk_start;
    chunk_start_adj(2:end) = chunk_start_adj(2:end) - chunk_pad;
    chunk_end_adj = chunk_end;
    chunk_end_adj(1:(nchunks-1)) = chunk_end_adj(1:(nchunks-1)) + chunk_pad;
end

chunk_ranges = [chunk_start' chunk_end' chunk_start_adj' chunk_end_adj'];
chunk_ranges(:,3) = max(chunk_ranges(:,3),1);
chunk_ranges(:,4) = min(chunk_ranges(:,4),total_images);

end


function I_raw = apply_intensity_adjustments_tile(I_raw, params, adj_ref,z_range)
% Apply intensity adjustments prior to aligning images

% Crop y_adj and flatfield to match reference
params.y_adj = crop_to_ref(adj_ref.y_adj,params.y_adj);
params.flatfield = crop_to_ref(adj_ref.flatfield,params.flatfield);
params.darkfield = crop_to_ref(adj_ref.darkfield,params.darkfield);

% Adjust intensities
for j = z_range
   if sum(I_raw(:,:,j) == 0)
       continue
   end
   if isequal(params.adjust_tile_shading,'basic')
       I_raw(:,:,j) = apply_intensity_adjustment(I_raw(:,:,j),...
           'flatfield', params.flatfield,...
           'darkfield', params.darkfield);
   elseif isequal(params.adjust_tile_shading,'manual')
      I_raw(:,:,j) = apply_intensity_adjustment(I_raw(:,:,j),...
          'y_adj',params.y_adj);
   end
end

end
