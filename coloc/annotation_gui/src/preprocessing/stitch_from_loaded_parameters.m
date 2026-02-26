function stitch_from_loaded_parameters(path_table, h_stitch_tforms, v_stitch_tforms, config)
% Stitch using previously calculated translations

tile_int_adjust = 50;      % Percentile to adjust tile intensity by slice if adj_params present
par_thresh = 20;           % Limit on number of slices before starting using parpool

% Stitch images using previously calculated parameters
fprintf('%s\t Begin stitching \n',datetime('now'))

% Create directory for stitched images
if ~isfolder(fullfile(config.output_directory,'stitched')) &&...
        isequal(config.save_images,"true")
    mkdir(fullfile(config.output_directory,'stitched'))
end

% Check if only certain channels to be stitched
if isempty(config.stitch_sub_channel)
    config.stitch_sub_channel = 1:length(config.markers);
end
if isequal(config.save_images,"false")
    config.stitch_sub_channel = 1;
end
%path_table = path_table(ismember(path_table.channel_num,config.stitch_sub_channel),:);

if ~isempty(h_stitch_tforms)
    total_sections = size(h_stitch_tforms,2);
elseif ~isempty(v_stitch_tforms)
    total_sections = size(v_stitch_tforms,2);
end

% Generate image name grid
img_name_grid = cell(max(path_table.y),max(path_table.x),...
    length(unique(path_table.markers)),total_sections);
path_table = sortrows(path_table,["z_adj","channel_num","x","y"],'ascend');

% Check if only certain sub-section is to be stitched
if ~isempty(config.stitch_sub_stack)
    z_range = config.stitch_sub_stack;
    path_table = path_table(ismember(path_table.z_adj,z_range),:);    
    img_name_grid = img_name_grid(:,:,:,z_range);    
else
    z_range = 1:size(img_name_grid,4);
end

% Arrange images into correct positions
try
    img_name_grid = reshape(path_table.file,size(img_name_grid));
catch ME
    if isequal(ME.identifier,'MATLAB:getReshapeDims:notSameNumel')
        error("Inconsistent image file information. Check configuration or " +....
            "recalculate stitching parameters and z_adjustment.")
    end
end

% Count number of sections
nb_sections = size(img_name_grid,4);

% Check if intensity adjustments are specified
if isequal(config.adjust_intensity,"true")
    if ~isfield(config,'adj_params')
        error("Intensity adjustments requested but could not locate adjustment parameters.")
    end
    adj_params = cell(1,length(config.markers));
    for i = 1:length(config.markers)
        adj_params{i} = config.adj_params.(config.markers(i));
    end
    fprintf('%s\t Applying some intensity adjustments \n',datetime('now'))
else
    fprintf('%s\t Stitching without intensity adjustments \n',datetime('now'))
    adj_params = [];
end

% Check for alignment table
if isequal(config.load_alignment_params,"true") && ~isempty(config.alignment_table)
    fprintf('%s\t Applying channel alignment parameters during stitching \n',datetime('now'))
end

% Create .mat file for stitching information
stitch_file = fullfile(config.output_directory,'variables','stitch_tforms.mat');
if ~isfile(stitch_file)
    error("Could not locate stitch parameters");    
end

% Start parallel pool
try
    p = gcp('nocreate');
catch
    p = 1;
end

if isempty(p) && nb_sections > par_thresh
    parpool
    p = 0;
elseif nb_sections < par_thresh
    p = 1;
elseif ~isempty(p)
    p = 0;
end

% Begin stitching
if p == 0
    parfor i = 1:length(z_range)
        % Print image being stitched
        fprintf('%s\t Stitching image %d \n',datetime('now'),z_range(i));
        stitch_worker_loaded(img_name_grid(:,:,:,i),h_stitch_tforms(:,z_range(i)),...
            v_stitch_tforms(:,z_range(i)),config,z_range(i),adj_params,tile_int_adjust);
    end
else
    for i = 1:length(z_range)
        % Print image being stitched
        fprintf('%s\t Stitching image %d \n',datetime('now'),z_range(i));
        stitch_worker_loaded(img_name_grid(:,:,:,i),h_stitch_tforms(:,z_range(i)),...
            v_stitch_tforms(:,z_range(i)),config,z_range(i),adj_params,tile_int_adjust);
    end
end
end

function stitch_worker_loaded(img_grid,h_tforms,v_tforms, config, z_idx, adj_params,tile_int)
% Stitch image grid

% Border padding amount along the edge to compensate for empty space left
% after alignemnt
border_pad = config.border_pad;

% Read images, adjust intensities, apply translations for multichannel
A = read_stitching_grid(img_grid,config.stitch_sub_channel,config.markers,...
    adj_params,config.alignment_table);

% Image grid info
[nrows,ncols,nchannels] = size(A);
[img_height,img_width] = size(A{1});

% Calculate overlaps in pixels
h_overlap = ceil(img_width*config.overlap);
v_overlap = ceil(img_height*config.overlap);
x0 = ceil(h_overlap/2);
y0 = ceil(v_overlap/2);

overlap_h_min = 1:h_overlap;
overlap_v_min = 1:v_overlap;

% Sizes of optimal, fully stitched image
full_width = img_width*ncols-h_overlap*(ncols-1);
full_height = img_height*nrows-v_overlap*(nrows-1);

%Save first column before horizontal stitching
B = A(:,1,:);

% Apply Horizontal Translations
a = 1;
for i = 1:nrows
    if ncols == 1
        continue
    end
    for j = 1:ncols-1
        % Update overlap region of the left image 
        overlap_h_max = size(B{i,1},2)-length(overlap_h_min)+1:size(B{i,1},2);
        
        % Load stitch parameters
        final_tform = affine2d([1 0 0; 0 1 0; h_tforms(a) h_tforms(a+1) 1]);
        
        % Create adjusted blending weights
        x1 = x0 - final_tform.T(3);
        if any(arrayfun(@(s) isequal(s,"sigmoid"),config.blending_method))
            w_h_adj = 1-1./(1 + exp(config.sd*(overlap_h_min-x1)));
        else
            w_h_adj = overlap_h_min/h_overlap;
        end
        
        % Clip ends based on horizontal translation where image intensity
        % is 0
        if final_tform.T(3) > 0
            adj_left = max(border_pad,ceil(final_tform.T(3)));
            w_h_adj(1:adj_left) = 0;
        else
            adj_right = max(border_pad,ceil(abs(final_tform.T(3))));
            w_h_adj(1:adj_right) = 0;
        end
        
        % Rescale non-cropped areas
        h_idx = w_h_adj>0 & w_h_adj<1;
        w_h_adj(h_idx) = (w_h_adj(h_idx) - min(w_h_adj(h_idx)))/(max(w_h_adj(h_idx)) - min(w_h_adj(h_idx)));

        % Transform and merge images (faster to use for loop on each channel)
        ref_fixed2 = imref2d([img_height img_width+floor(final_tform.T(3))]);
        for k = 1:nchannels
            reg_img = imwarp(A{i,j+1,k},final_tform,'OutputView',ref_fixed2,'FillValues',0);
            c_idx = config.stitch_sub_channel(k);
            
            %Adjust intensity again?
            %if isequal(config.adjust_tile_position(c_idx),"true")
            %    reg_img = tile_pair_adjustment(B{i,k}(:,overlap_h_max),...
            %        reg_img(:,overlap_h_min),reg_img,tile_int,true);
            %end
            B{i,k} = blend_images(reg_img,B{i,k},false,config.blending_method(c_idx),w_h_adj);
        end
        a = a+2;
    end
end

%Crop horizontally stitched images to minimum width
min_width = min(cellfun(@(s) size(s,2),B(:,1)));
B = cellfun(@(s) s(:,1:min_width), B,'UniformOutput',false);
I = B(1,:);

% Apply Vertical Translations
b = 1;
for i = 1:length(B)-1
    % Update overlap region of the top image 
    overlap_v_max = size(I{1},1)-length(overlap_v_min)+1:size(I{1},1);
    
    % Load stitch parameters
    final_tform = affine2d([1 0 0; 0 1 0; v_tforms(b) v_tforms(b+1) 1]);
    
    % Create adjusted blending weights
    y1 = y0 + final_tform.T(6);
    
    if isequal(config.blending_method(k),"sigmoid")
        w_v_adj = 1-1./(1 + exp(config.sd*(overlap_v_min-y1)))';
    else
        w_v_adj = (overlap_v_min/v_overlap)';
    end
    
    % Clip ends based on vertical translation where image intensity
    % is 0
    if final_tform.T(6) > 0
        adj_top = max(border_pad,ceil(final_tform.T(6)));
        w_v_adj(1:adj_top) = 0;
    else
        adj_bottom = max(border_pad,ceil(abs(final_tform.T(6))));
        w_v_adj(1:adj_bottom) = 0;
    end
    
    % Rescale non-cropped areas
    v_idx = w_v_adj>0 & w_v_adj<1;
    w_v_adj(v_idx) = (w_v_adj(v_idx) - min(w_v_adj(v_idx)))/(max(w_v_adj(v_idx)) - min(w_v_adj(v_idx)));
    
    %Transform images
    ref_fixed2 = imref2d([img_height+floor(final_tform.T(6)) size(I{1},2)]);
    
    for k = 1:nchannels
        reg_img = imwarp(B{i+1,k},final_tform,'OutputView',ref_fixed2,'FillValues',0);
        c_idx = config.stitch_sub_channel(k);
        
        %Adjust intensity again?
        %if isequal(config.adjust_tile_position(c_idx),"true")
        %    reg_img = tile_pair_adjustment(I{k}(overlap_v_max,:),reg_img(overlap_v_min,:),...
        %        reg_img,tile_int,true);
        %end
        I{k} = blend_images(reg_img,I{k},false,config.blending_method(c_idx),w_v_adj); 
    end
    b = b+2;
end

% Postprocess the image with various filters, background subtraction, etc.
c_idx = config.stitch_sub_channel;
for i = 1:length(c_idx)
    I{i} = postprocess_image(config, I{i}, c_idx(i));
end

%Crop or pad images based on ideal size
%I = cellfun(@(s) crop_to_ref(zeros(full_height,full_width),s),I,'UniformOutput',false);
I = cellfun(@(s) uint16(crop_to_ref(zeros(full_height+border_pad,full_width+border_pad),s)),I,'UniformOutput',false);


%Save images as individual channels (will be large)
for i = 1:length(c_idx)
    img_name = sprintf('%s_%s_C%d_%s_stitched.tif',...
        config.sample_id,num2str(z_idx,'%04.f'),c_idx(i),config.markers(c_idx(i)));
    img_path = fullfile(char(config.output_directory),'stitched',img_name);
    imwrite(uint16(I{i}),img_path)
end

end
