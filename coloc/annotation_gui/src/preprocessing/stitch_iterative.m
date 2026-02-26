function I = stitch_iterative(config, path_table, z_slice, pseudo)
%--------------------------------------------------------------------------
% Perform iterative 2D image stitching using phase correlation with
% optional refinement using SIFT. Image slices positions are presumed to be
% adjusted along the z dimension to find the optimal correspondence.
% Therefore this program will start from roughly the middle z positions and
% work its way out towards the first and last slice.
%--------------------------------------------------------------------------

% Defaults
defaults.usfac = 10;                        % Precision of translations
defaults.peaks = 5;                         % Number of phase correlation peaks to test
defaults.border_pad = config.border_pad;    % Border cropping along edges
defaults.min_overlap = 1;                   % Do not adjust for now, does not mesh with limit_xy. Minimum overlapping region in pixels
defaults.signal_thresh = 0.05;              % Minimum fraction of poisitive pixels
defaults.limit_xy = 3;                      % Maximum shift in either x or y from previous slice
defaults.tile_int_adjust = 50;              % Percentile to adjust tile intensity by slice if adj_params present
defaults.pseudo = false;                    % Pseudocolroed image for testing

% If stitching single slice, add to config
if nargin >2
    if length(z_slice) >1
        error("Can only specify 1 z position to stitch")
    end
    config.stitch_start_slice = z_slice;
    config.stitch_sub_stack = z_slice;
end

if nargin >3 
    defaults.pseudo = pseudo;
end
var_dir = fullfile(config.output_directory,'variables');

% Create directory for stitched images
if ~isfolder(fullfile(config.output_directory,'stitched')) &&...
        isequal(config.save_images,"true") &&...
        ~defaults.pseudo
    mkdir(fullfile(config.output_directory,'stitched'))
end

% If using SIFT, check if vl_feat toolbox exists
if isequal(config.sift_refinement,"true")
    if exist('vl_sift','file') == 0
        error("Could not call vl_feat toolbox. Download here: https://www.vlfeat.org/install-matlab.html")
    else
        fprintf("%s\t Using SIFT refinement \n",datetime('now'))
    end
end

% Check if only certain channels to be stitched
if isempty(config.stitch_sub_channel)
    config.stitch_sub_channel = 1:length(config.markers);
end
if isequal(config.save_images,"false")
    config.stitch_sub_channel = 1;
end
%path_table = path_table(ismember(path_table.channel_num,config.stitch_sub_channel),:);

% Get rows/columns and adjust z
nrows = length(unique(path_table.y));
ncols = length(unique(path_table.x));
if ~any(ismember(path_table.Properties.VariableNames,{'z_adj'}))
    % Load z displacement from file
    var_file = fullfile(var_dir,'z_disp_matrix.mat');
    load(var_file, 'z_disp_matrix')
    assert(nrows == size(z_disp_matrix,1) && ncols == size(z_disp_matrix,2), "Loaded z adjustment parameters do not match number of tiles "+...
        "detected in the input image directory. Check input image directory or update z adjustment")
    path_table = apply_adjusted_z(path_table, z_disp_matrix);
end

% Generate image name grid
img_name_grid = cell(max(path_table.y),max(path_table.x),length(unique(path_table.markers)),max(path_table.z_adj));
path_table = sortrows(path_table,["z_adj","channel_num","x","y"],'ascend');

% Arrange images into correct positions
try
    img_name_grid = reshape(path_table.file,size(img_name_grid));
catch ME
    if isequal(ME.identifier,'MATLAB:getReshapeDims:notSameNumel')
        error("Inconsistent image file information. Recalculate adjusted z and/or check configuration")
    end
end

% Count number of sections
nb_sections = size(img_name_grid,4);

% Check if only certain sub-section is to be stitched
if ~isempty(config.stitch_sub_stack)
    img_name_grid = img_name_grid(:,:,:,config.stitch_sub_stack);
    z_range = config.stitch_sub_stack;
    nb_sections = length(z_range);
else
    z_range = 1:size(img_name_grid,4);
end

% Check if intensity adjustments are specified
if isequal(config.adjust_intensity,"true")
    if ~isfield(config,'adj_params')
        try
            load(fullfile(var_dir,'adj_params.mat'),'adj_params')
            [adj_params, config] = check_adj_parameters(adj_params,config,nrows,ncols);
            config.adj_params = adj_params;
        catch
            error("Intensity adjustments requested but could not locate adjustment parameters.")
        end
    end
    adj_params = cell(1,length(config.markers));
    for i = 1:length(config.markers)
        adj_params{i} = config.adj_params.(config.markers(i));
    end
    fprintf('%s\t Applying some intensity adjustments \n',datetime('now'))
else
    fprintf('%s\t Stitching without intensity adjustments \n',datetime('now'))
    adj_params = [];
    if isempty(config.signalThresh)
        config.signalThresh = 0;
    end
end

% Check for alignment table
if isequal(config.load_alignment_params,"true")
    if ~isfield(config,'alignment_table')
        load(fullfile(var_dir,'alignment_table.mat'),'alignment_table')
        config.alignment_table = alignment_table;
    end
    fprintf('%s\t Applying channel alignment parameters during stitching \n',datetime('now'))
end

% Create .mat file for storing stitching information
stitch_file = fullfile(config.output_directory,'variables','stitch_tforms.mat');
if ~isfile(stitch_file) && ~defaults.pseudo
    error("Could not locate stitch parameters");    
end

% Determine order of images being stitched based on which z position has
% the most bright features in the tile with lowest signal. Otherwise start
% from middle or from user-specified z 
if isnumeric(config.stitch_start_slice) && ~isempty(config.stitch_start_slice)
    fprintf("%s\t Using defined stitching start position \n",datetime('now'))

    % Start from specified z position
    start_z = config.stitch_start_slice;
    start_z = ceil(start_z * nb_sections/max(path_table.z));
    
elseif isempty(config.stitch_start_slice)
    fprintf("%s\t Calculating stitching start position \n",datetime('now'))
    % Check 5% of images
    pos = max(1,floor(linspace(1,nb_sections,ceil(nb_sections*0.05)))); 
    signal = zeros(length(pos),numel(img_name_grid(:,:,1)));
    for i = 1:length(pos)
        img_grid = img_name_grid(:,:,1,pos(i));
        for j = 1:numel(img_grid)
            tempI = imread(img_grid{j});
            if size(tempI,3)>1
                tempI = tempI(:,:,1);
            end
            signal(i,j) = sum(tempI(:)>config.signalThresh(1)*65535)/numel(tempI);
        end
    end
    % Find which tile has the lowest mean signal 
    signal_ave = mean(signal,1);
    [~,z_idx] = max(signal(:,signal_ave == min(signal_ave)));
    start_z = pos(z_idx);  
    
else
    fprintf("%s\t Stitching from middle of stack \n",datetime('now'))
    
    % Otherwise start from the middle
    start_z = ceil(size(img_name_grid,4)/2);
end 

if start_z == nb_sections
    start_z = max(1,start_z-1);
end

% If stitching single image, return image without saving
if nargin>2
    [~,~,I]=stitch_worker(img_name_grid(:,:,:,1),[],[],...
            config,z_range,adj_params,defaults);
    if length(I) == 1
        I = I{1};
    end
    return
end

% Create stitching folder
save_dir = fullfile(config.output_directory,'stitched');
if ~isfolder(save_dir)
    mkdir(save_dir)
end

% Start parallel pool
try
    p = gcp('nocreate');
    max_workers = 2;
    if isempty(p) && nb_sections > 20
        parpool(2)
    elseif nb_sections <= 20
        max_workers = 0;
    end
catch
    max_workers = 0;
end

fprintf("%s\t Begin stitching %d slices \n",datetime('now'),length(z_range))

%Begin stitching from the starting position. Stitching proceeds iteratively
%from here to top and bottom. Previous translations are used as thresholds
%for maximum translations for the current section. If threshold is exceeded
%(i.e. images are moving more than expected) the previous translation is
%used as the current section is lacking enough features (likely because
%it's at the edge of the sample
%for idx2 = 1:2
%parfor (idx2 = 1:min(2,nb_sections),max_workers)
for idx2 = 1:min(2,nb_sections)
    m = matfile(stitch_file,'Writable',true);
    if ~isempty(config.stitch_sub_stack) && ~isequal(config.stitch_images,'update')
       if idx2 == 1 
            h_tform = m.h_stitch_tforms(:,z_range(start_z));
            v_tform = m.v_stitch_tforms(:,z_range(start_z));
       else
            h_tform = m.h_stitch_tforms(:,z_range(start_z+1));
            v_tform = m.v_stitch_tforms(:,z_range(start_z+1));
       end
       if all(h_tform == 0) && all(v_tform == 0)
           h_tform = []; v_tform = [];
       else
           h_tform = reshape(num2cell(reshape(h_tform,2,length(h_tform)/2),1),nrows,ncols-1);
           v_tform = reshape(num2cell(reshape(v_tform,2,length(v_tform)/2),1),nrows-1,1);
       end
    else
        h_tform = []; v_tform = [];
    end
    % Split into 2 workers
    if idx2 == 1
        % From middle to top
        for i = fliplr(1:start_z)
            z_pos = z_range(i);
            
            % Reset previous translations
            pre_h_tform = h_tform;
            pre_v_tform = v_tform;
            
            % Print image being stitched
            fprintf('%s\t Stitching image %d \n',datetime('now'),z_pos);
            [h_tform,v_tform]=stitch_worker(img_name_grid(:,:,:,i),pre_h_tform,pre_v_tform,...
                config,z_pos,adj_params,defaults);
            
            % Store translations
            if ~isempty(h_tform)
                h_tform1 = h_tform';
                m.h_stitch_tforms(:,z_pos) = [h_tform1{:}]'; 
            end
            if ~isempty(v_tform)
                v_tform1 = v_tform';
                m.v_stitch_tforms(:,z_pos) = [v_tform1{:}]';
            end
        end
    else
        %From middle to bottom
        for j = start_z+1:nb_sections
            z_pos = z_range(j);
            
            % Reset previous translations
            pre_h_tform = h_tform;
            pre_v_tform = v_tform;

            % Print image being stitched
            fprintf('%s\t Stitching image %d \n',datetime('now'),z_pos);
            [h_tform,v_tform]=stitch_worker(img_name_grid(:,:,:,j),pre_h_tform,pre_v_tform,...
                config,z_pos,adj_params,defaults);
           
            % Store translations
            if ~isempty(h_tform)
                h_tform1 = h_tform';
                m.h_stitch_tforms(:,z_pos) = [h_tform1{:}]'; 
            end
            if ~isempty(v_tform)
                v_tform1 = v_tform';
                m.v_stitch_tforms(:,z_pos) = [v_tform1{:}]';
            end
        end
    end
end

end

function [pre_h_tform,pre_v_tform,I] = stitch_worker(img_grid,pre_h_tform,pre_v_tform,config,z_idx,adj_params,defaults)
% Worker for stitching_iterative function

% Defaults
usfac = defaults.usfac;
peaks = defaults.peaks;
border_pad = defaults.border_pad;
min_overlap = defaults.min_overlap;
signal_thresh = defaults.signal_thresh;
limit_xy = defaults.limit_xy;
tile_int = defaults.tile_int_adjust; 
pseudo = defaults.pseudo;

% Option for testing. Convert blending to pseudocolored image
testing = false;
if nargout == 3
    testing = true;    
end

% Whether params are being updated
update = false;
if isequal(config.stitch_images,"update") && ~isempty(config.stitch_sub_stack)
    update = true;
end

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

% Calculate extended overlap if presumed overlap is small
if h_overlap < min_overlap
    ext_adj_h = min([min_overlap, img_width]) - h_overlap;
else
    ext_adj_h = 0;
end
if v_overlap < min_overlap
    ext_adj_v = min([min_overlap, img_height]) - v_overlap;
else
    ext_adj_v = 0;
end
overlap_h_min = 1:h_overlap + ext_adj_h;
overlap_v_min = 1:v_overlap + ext_adj_v;

% Sizes of optimal, fully stitched image
full_width = img_width*ncols-h_overlap*(ncols-1);
full_height = img_height*nrows-v_overlap*(nrows-1);

% Check for previous translations and set limits
if ~iscell(pre_h_tform)
    limit_xy = NaN;
    pre_h_tform = repmat({[NaN,NaN]},[nrows,ncols-1]);
    shift_threshold = repmat({NaN},[nrows,ncols-1]);
    first_z = "true";
else
    shift_threshold = cellfun(@(s) max(abs(s))+limit_xy,pre_h_tform,'UniformOutput',false);
    first_z = "false";
end

%Save first column before horizontal stitching
B = A(:,1,:);

% Calculate horizontal translations
for i = 1:nrows 
    if ncols == 1
        pre_h_tform = {};
        continue
    end
    for j = 1:ncols-1
        % Update overlap region of the left image 
        overlap_h_max = size(B{i,1},2)-length(overlap_h_min)+1:size(B{i,1},2);

        % Load overlapped regions
        ref_img = B{i,:,1}(:,overlap_h_max);
        mov_img = A{i,j+1,1}(:,overlap_h_min);

        % Check number of bright pixels
        signal = sum(ref_img(:)>config.signalThresh(1)*65535)/numel(mov_img);
        if signal<signal_thresh
            if isequal(first_z, "true")
                % If first z and low intensity, use previous horizontal
                % tform
                final_tform = final_tform;
                disp(i);disp(j)
            else
                try 
                    final_tform = affine2d([1 0 0; 0 1 0; pre_h_tform{i,j}(1) pre_h_tform{i,j}(2) 1]);
                catch
                    if isnan(pre_h_tform{i,j}(1)) || isnan(pre_h_tform{i,j}(2))
                        error("Calling NaN as expected transform. Check thresholds or recalculate start z position.")
                    end
                end
            end
        else
            % Perform phase correlation
            [pc_img,ref_img, tformPC] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold{i,j});

            % Use previous translation if shift is large
            if isempty(tformPC) ||...
                    abs(tformPC.T(3)-pre_h_tform{i,j}(1))>limit_xy+ext_adj_h ||...
                    abs(tformPC.T(6)-pre_h_tform{i,j}(2))>limit_xy
                tformPC = affine2d([1 0 0; 0 1 0; pre_h_tform{i,j}(1) pre_h_tform{i,j}(2) 1]);
                pc_img = imtranslate(mov_img, [pre_h_tform{i,j}(1) pre_h_tform{i,j}(2)]);
            end

            % Refine using SIFT
            if isequal(config.sift_refinement,'true')
                tformSIFT = sift_refinement_worker(pc_img,ref_img);
                final_tform = affine2d(tformPC.T*tformSIFT.T);
            else
                final_tform = tformPC;
            end

            % If not able to calculate transform, use previous transform
            if ~update &&...
                    (abs(final_tform.T(3)-pre_h_tform{i,j}(1))>limit_xy+ext_adj_h ||...
                    abs(final_tform.T(6)-pre_h_tform{i,j}(2))>limit_xy)
                final_tform = affine2d([1 0 0; 0 1 0; pre_h_tform{i,j}(1) pre_h_tform{i,j}(2) 1]);
                if signal > signal_thresh*5
                    fprintf('%s\t Warning: large horizontal displacement at %d x %d \n',...
                        datetime('now'),i,j);                
                end
            end
        end
        
        % If overlap is extended, resize to to expected overlapping region
        if ext_adj_h > 0
            final_tform.T(3) = final_tform.T(3) - ext_adj_h;
            overlap_h_min1 = overlap_h_min(1:h_overlap);
        else
            overlap_h_min1 = overlap_h_min;
        end
        
        % Create adjusted blending weights
        x1 = x0 + final_tform.T(3);
        if any(arrayfun(@(s) isequal(s,"sigmoid"),config.blending_method))
            w_h_adj = 1-1./(1 + exp(config.sd*(overlap_h_min1-x1)));
        else
            w_h_adj = overlap_h_min1/h_overlap;
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
        
        % Save translation        
        pre_h_tform{i,j} = [final_tform.T(3), final_tform.T(6)];
                
        % Transform and merge images (faster to use for loop on each channel)
        ref_fixed2 = imref2d([img_height img_width+floor(final_tform.T(3))]);
        for k = 1:nchannels
            reg_img = imwarp(A{i,j+1,k},final_tform,'OutputView',ref_fixed2,'FillValues',0);
            c_idx = config.stitch_sub_channel(k);
            
            %Adjust intensity again? Not stable in images with few features
            %if isequal(config.adjust_tile_position(c_idx),"true")
            %    reg_img = tile_pair_adjustment(B{i,k}(:,overlap_h_max),...
            %        reg_img(:,overlap_h_min),reg_img,tile_int);
            %end            
            B{i,k} = blend_images(reg_img,B{i,k},false,config.blending_method(c_idx),w_h_adj);  
        end
        
    end
end  

% Crop horizontally stitched images to minimum width
min_width = min(cellfun(@(s) size(s,2),B(:,1)));
B = cellfun(@(s) s(:,1:min_width), B,'UniformOutput',false);

% Identify horizontally stitched images minimum height
if ncols == 1
    min_height = zeros(nrows,1);
    max_height = zeros(nrows,1);
else
    min_height = ceil(abs(cellfun(@(s) max(0,abs(min(s(:,2)))),pre_h_tform)));
    max_height = ceil(abs(cellfun(@(s) max(0,max(s(:,2))),pre_h_tform)));
end
I = B(1,:);

% Check for previous translations and set limits
if ~iscell(pre_v_tform)
    limit_xy = NaN;
    pre_v_tform = repmat({[NaN,NaN]},[1 size(B,1)-1]);
    shift_threshold = repmat({NaN},[1 size(B,1)-1]);
else
    shift_threshold = cellfun(@(s) max(abs(s))+limit_xy, pre_v_tform,'UniformOutput',false);
end

% Do blended row tiles
for i = 1:size(B,1)-1
    % Update overlap region of the top image 
    overlap_v_max = size(I{1},1)-length(overlap_v_min)+1:size(I{1},1);
    
    % Load overlapped regions
    ref_img = I{1}(overlap_v_max,1:min_width);
    if min_height(i)>0
        ref_img(end-min_height(i):end,:) = 0;
        %I{:}(end-min_height(i)+1:end,:) = 0;
    end

    mov_img = B{i+1,1}(overlap_v_min,1:min_width);
    if length(max_height) > 1 && max_height(i+1)>0
        mov_img(1:max_height(i),:) = 0;
        %B{i+1,:}(1:max_height(i)-1,:) = 0;
    end
    
    % When there is no intensity, use previous translation
    signal = sum(ref_img(:)>config.signalThresh(1)*65535)/numel(mov_img);
    if signal<signal_thresh
        final_tform = affine2d([1 0 0; 0 1 0; pre_v_tform{i}(1) pre_v_tform{i}(2) 1]);
    else
        % Perform phase correlation and refine with SIFT
        [pc_img,ref_img,tformPC] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold{i});

        if isempty(tformPC) ||...
                abs(tformPC.T(3)-pre_v_tform{i}(1))>limit_xy ||...
                abs(tformPC.T(6)-pre_v_tform{i}(2))>limit_xy+ext_adj_v
            tformPC = affine2d([1 0 0; 0 1 0; pre_v_tform{i}(1) pre_v_tform{i}(2) 1]);
            pc_img = imtranslate(mov_img,[pre_v_tform{i}(1) pre_v_tform{i}(2)]);
        end
        
        %Refine using SIFT
        if isequal(config.sift_refinement,'true')
            [tformSIFT] = sift_refinement_worker(pc_img,ref_img);
            final_tform = affine2d(tformPC.T*tformSIFT.T);
        else
            final_tform = tformPC;
        end

        %If not able to calculate transform, use previous transform
        if ~update &&...
                (abs(final_tform.T(3)-pre_v_tform{i}(1))>limit_xy ||...
                abs(final_tform.T(6)-pre_v_tform{i}(2))>limit_xy+ext_adj_v)
            final_tform = affine2d([1 0 0; 0 1 0; pre_v_tform{i}(1) pre_v_tform{i}(2) 1]);
            if signal>signal_thresh*5
                fprintf('%s\t Warning: large vertical displacement at %d\n',datetime('now'),i);                            
            end
        end
    end
    
    % If overlap is extended, resize to to expected overlapping region
    if ext_adj_v > 0
        final_tform.T(6) = final_tform.T(6) - ext_adj_v;
        overlap_v_min1 = overlap_v_min(1:v_overlap);
    else
        overlap_v_min1 = overlap_v_min;
    end
    
    % Create adjusted blending weights
    y1 = y0 + final_tform.T(6);
    if isequal(config.blending_method,"sigmoid")
        w_v_adj = 1-1./(1 + exp(config.sd*(overlap_v_min1-y1)))';
    else
        w_v_adj = (overlap_v_min1/v_overlap)';
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
        
        %Adjust intensity again? Not stable in images with few features
        %if isequal(config.adjust_tile_position(c_idx),"true")
        %    reg_img = tile_pair_adjustment(I{k}(overlap_v_max,:),reg_img(overlap_v_min,:),...
        %        reg_img,tile_int);
        %end
        I{k} = blend_images(reg_img,I{k},false,config.blending_method(c_idx),w_v_adj); 
    end
    
    %Save translation
    pre_v_tform{i} = [final_tform.T(3), final_tform.T(6)];
end

% Return if only calculating parameters
if isequal(config.save_images,'false') && ~testing
    return
end

% Postprocess the image with various filters, background subtraction, etc.
c_idx = config.stitch_sub_channel;
for i = 1:length(c_idx)
    I{i} = postprocess_image(config, I{i}, c_idx(i));
end

% Crop or pad images based on ideal size
I = cellfun(@(s) uint16(crop_to_ref(zeros(full_height+border_pad,full_width+border_pad),s)),I,'UniformOutput',false);
%I = cellfun(@(s) uint16(s),I,'UniformOutput',false);


% Return if only calculating parameters
if ~testing
    % Save images as individual channels (will be large)
    for i = 1:length(c_idx)
        img_name = sprintf('%s_%s_C%d_%s_stitched.tif',...
            config.sample_id,num2str(z_idx,'%04.f'),c_idx(i),config.markers(c_idx(i)));
        img_path = fullfile(char(config.output_directory),'stitched',img_name);
        imwrite(I{i},img_path)
    end
    clear I
end

end

