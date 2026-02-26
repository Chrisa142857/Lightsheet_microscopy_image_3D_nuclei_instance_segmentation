function coreg_table = align_by_translation(config,path_table,z_displacement_align,only_pc)
%--------------------------------------------------------------------------
% Align image channels by translating 2D slices. Requires a z_displacement
% structure to pre-align slices along z. Phase correlation is used to get a
% rough alignment and actual registration is performed using MATLAB's 
% imregister function (Image Processing Toolbox).
%--------------------------------------------------------------------------
% Usage:
% coreg_table = 
% align_by_translation(config,path_table,z_displacement_align, only_pc)
%
%--------------------------------------------------------------------------
% Inputs:
% config: Config structure from NM_process.
%
% path_table: Path table containing only 1 tile position that will be
% aligned.
%
% z_displacement_align: Structure containing z displacement matrix for 
% markers to align. Otherwise z displacement is assumed to be 0 for all 
% markers.
%
% only_pc: (logical) Use only phase correlation for alignment without
% perfmorming registration. (default: false)
%
%--------------------------------------------------------------------------
% Outputs:
% coreg_table: Table containing aligned filenames and their respective
% translations for this tile.
%
%--------------------------------------------------------------------------

% Default displacement threshold for comparing to previous translation in
% pixels.
shift_threshold = 3;    % Max shift difference between z positions
max_shift = 100;        % Max overall shift thershold for pc
min_signal = 0.005;     % Minimum fraction of signal pixels for registering
peaks = 3;              % Phase correlation peaks to test
usfac = 10;              % Precision of phase correlation

% Define intensity-based registration settings
% Use optimizer and metric setting
metric = registration.metric.MattesMutualInformation;
optimizer = registration.optimizer.RegularStepGradientDescent;
       
metric.NumberOfSpatialSamples = 200;
metric.NumberOfHistogramBins = 50;
       
optimizer.GradientMagnitudeTolerance = 1.00000e-04;
optimizer.MinimumStepLength = 1.00000e-05;
optimizer.MaximumStepLength = 1.00000e-01;
optimizer.MaximumIterations = 50;
optimizer.RelaxationFactor = 0.5;

% Use only phase correlation
if nargin<4
    if isequal(config.only_pc,"true")
        only_pc = true;
    else
        only_pc = false;
    end
end

% Unpack variables from config structure
signalThresh = config.signalThresh;
align_stepsize = config.align_stepsize;
save_images = config.save_images;

align_slices = config.align_slices;
align_channels = config.align_channels;

% Get relevant tile information
channel_num = unique(path_table.channel_num);
markers = config.markers(channel_num);
col = unique(path_table.x); 
row = unique(path_table.y);
ref_slices = unique(path_table.z);

% Make sure there's only 1 tile
assert(length(row) == 1 & length(col) ==1 ,...
    "Image path table should contain only 1 unique tile position")

% Subset channels if not doing all 
if ~isempty(align_channels)
    assert(all(ismember(markers(align_channels),markers)),"Align channels "+...
        "are outside of range for the number of markers present")
    align_markers = markers(align_channels);
    %path_table = path_table(ismember(path_table.markers, align_markers),:);
else
    align_channels = 1:length(markers);
    align_markers = markers;
end

% If only aligning certain channels or slices, load the alignment table
update_table = true;
if isequal(config.channel_alignment,"true")
    save_path = fullfile(config.output_directory,'variables','alignment_table.mat');
    if isfile(save_path)
        load(save_path,'alignment_table')
        alignment_table = alignment_table{row,col};
        if ~isempty(alignment_table)
            update_table = false;
        end
        
        % Check height and width of loaded table. Otherwise clear table and
        % make a new one
        %if ~isempty(config.align_slices) || ~isempty(config.align_channels)
        %    assert(height(alignment_table) == max(ref_slices), "Loaded alignment table height "+...
        %        "does not match the number of z slices present.")
        %    assert(width(alignment_table) == length(markers) + (length(markers)-1)*5 + 1,"Number of "+...
        %        "channels in the alignment table does not match the number of channels in the input "+...
        %        "image directory.")
        %end
    end
end

% Adjust for resolution
res_adj = cell(1,length(align_markers)-1); res_equal = true(1,length(align_markers)-1);
for i = 2:length(markers)
    tempI = read_img(path_table,1);
    res_adj{i-1} = size(tempI)./(config.resolution{1}(1:2)./config.resolution{channel_num(i)}(1:2));
    res_equal(i-1) = all(config.resolution{1}(1:2) == config.resolution{i}(1:2));
end

% If not updating, save images and return
if ~update_table
    write_aligned_images(config,alignment_table,res_equal,res_adj,row,col)
    coreg_table = alignment_table;
   return 
end

% Get z displacement
z_displacement = zeros(1,length(markers));
if nargin >2 && ~isempty(z_displacement_align)
    for i = 1:length(align_markers)
        z_displacement(align_channels(i)) = z_displacement_align.(align_markers(i))(row,col);
        fprintf("%s\t Using z_displacement %d for marker %s\n", datetime('now'),...
            z_displacement(align_channels(i)),align_markers(i));
    end
end

% Create matrix with with z displacements for each channel
z_list = zeros(length(ref_slices),length(markers));
z_list(:,1) = ref_slices;

% Creat new table
path_new = path_table(path_table.markers == markers(1),:);
for i = 2:length(markers)  
    table_sub = path_table(path_table.markers == markers(i),:);
    z_list(:,i) = table_sub.z - z_displacement(i);
    table_sub.z = z_list(:,i);
    path_new = cat(1,path_new,table_sub);
end
path_full = path_new;

% Subset slices
if ~isempty(align_slices)    
    align_slices = [align_slices{:}];
    align_slices = align_slices(align_slices<=max(ref_slices));
    for i = 1:length(align_slices)
        path_new = path_new(ismember(path_new.z,align_slices),:);
    end
    min_z = min(align_slices);
    max_z = max(align_slices);
else
    % Take every n images along z if coreg_stepsize > 1 
    min_z = find(all(z_list>0,2),1);
    max_z = find(all(z_list<=max(z_list(:,1)),2),1,'last');
    if isempty(min_z) || isempty(max_z)
        error("Z positions are too far out of range for this image set. Try "+...
            "aligning each marker individually or set z_displacement to 0.")
    end
end

% Get subset z_positions
if align_stepsize > max(ref_slices)
    z = round(median(ref_slices));
elseif align_stepsize > 1
    z = min_z:align_stepsize:max_z;
else
    z = ref_slices;
end
path_new = path_new(ismember(path_new.z,z),:);

% Create a matrix for recording information. This info will get saved in
% structure array later
order_m = cat(1,z,zeros(1+5*(numel(markers)-1),length(z)));

% Read each image in stack and measure number of bright pixels
for i = 2:length(markers)
    for j = 1:length(z)
        mov_img = read_img(path_new,[i,z(j)]);        
        order_m(i,j) = numel(mov_img(mov_img>signalThresh(i)))/numel(mov_img);
    end
end

% Sort by average of bright pixels in all channels
sort_row = 1+numel(markers);
order_m(sort_row,:) = mean(order_m(2:length(markers),:),1);

% Sort the matrix to register brightest positions first
order_m = sortrows(order_m',sort_row,'descend');
reg_img = cell(1,length(markers)-1);

% Note: order_m matrix is as follow:
% ref_z position, signal of each channel, mean signal of all non-ref
% channels, registration type, x translation, y translation, cc. Reg. type
% through cc repeated for each channel.

% Run channel coregistration
for i = 1:size(order_m,1)
    % Get z position 
    z_idx = order_m(i,1);
   
    % Read reference image
    ref_img = read_img(path_new,[1,z_idx]);
   
    % For each non-reference channel, perform registration using translation
    for j = align_channels
        if order_m(i,j) > min_signal
            % Index within order matrix
            idx = 2+length(markers)+4*(j-2);

            % Read moving image
            mov_img = read_img(path_new,[j,z_idx]);
            
            % Continue if out of range
            if isempty(mov_img)
                continue
            end

            % Readjust resolution if necessary
            if ~res_equal(j-1)
                mov_img = imresize(mov_img,res_adj{j-1},'bicubic');
            end

           % Do simple crop in case image sizes don't match up
           if any(size(ref_img) ~= size(mov_img))
               mov_img = crop_to_ref(ref_img, mov_img);
           end

           % Register using phase correlation. Record 1,cross correlation,error
           [~,~,tform] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,max_shift);
                      
           % If tform is empty, use nearest translation
           if isempty(tform)
               m_subset = order_m(order_m(:,idx)~=0,:);
               [~,nearest_idx] = min(abs(m_subset(:,1)-z_idx));
               tform = affine2d([1 0 0; 0 1 0; 0 0 1]);
               tform.T(3) = m_subset(nearest_idx,idx+1);
               tform.T(6) = m_subset(nearest_idx,idx+2);
               order_m(i,idx) = 1; % Note poor pc result   
               if i == 1
                   warning("Poor phase correlation result for the initial slice. Alignment "+....
                       "not likely to succeed. Check images and parameters....\n")
                   pause(5)
               end
           end

           % Check for big translations in phase correlation. It's very important to use
           % a good initial translation before intensity-based registration
           if i > length(z)*0.1
               % Take median of translations calculated so far            
               x_med = median(order_m(1:i-1,idx+1));
               y_med = median(order_m(1:i-1,idx+2));

               % Calculate distance from median value
               d = ((tform.T(3)-x_med).^2 + (tform.T(6)-y_med).^2).^0.5;

               % If translation is too far from the median, take nearest
               % translation
               if d > shift_threshold
                   m_subset = order_m(order_m(:,idx)~=0,:);
                   [~,nearest_idx] = min(abs(m_subset(:,1)-z_idx));
                   tform.T(3) = m_subset(nearest_idx,idx+1);
                   tform.T(6) = m_subset(nearest_idx,idx+2);
                   order_m(i,idx)=1; % Note poor pc result  
               end
           end
           
           if ~only_pc
               % Use MATLAB's intensity-based registration to refine registration
               % Calculate transform
               tform = imregtform(mov_img,ref_img,'translation',optimizer,metric,...
                   'PyramidLevels',1,'InitialTransformation',tform);               
           end
           order_m(i,idx)=order_m(i,idx)+1; % Save translation method

           % Apply the transform
           reg_img{j-1} = imwarp(mov_img,tform,'OutputView',imref2d(size(ref_img)));
           
           % Save x,y translations
           order_m(i,idx+1)=tform.T(3);
           order_m(i,idx+2)=tform.T(6);

           % Calculate cross correlation
           order_m(i,idx+3) = calc_corr2(reg_img{j-1},ref_img);
       end
    end
    
   % Update status
   fprintf("%s\t %d out of %d images completed\n", datetime('now'),i,size(order_m,1));
end

% If stepsize > 1, interpolate values to get missing translations
[~,sort_idx] = sort(order_m(:,1));
order_m = order_m(sort_idx,:);

% Check for outliers and fill in 'no signal' images
for j = align_channels
    % Index within order matrix
    idx = 2+length(markers)+4*(j-2);
    
    % Check for any good translations
    if ~any(order_m(:,idx)>0)
        error("No valid registration results found in tile")
    end

    % Subset translations for this channel
    subset0 = order_m(:,idx:idx+2);
    
    % Remove rows not registered
    nonreg_idx = subset0(:,1)==0;
    reg_idx = subset0(:,1)>0;
    subset1 = subset0(reg_idx,:);
    
    % Detect and replace outlier using moving median sliding window.
    % Use 10 image sliding window
    window = min(10,length(subset1));
    subset1(:,2:3) = filloutliers(subset1(:,2:3),'linear','movmedian',window,1);
    
    % Replace outlier values in order_m
    order_m(reg_idx,idx:idx+2) = subset1;
    
    % Use nearest translation for images where registration failed
    nonreg_z = order_m(nonreg_idx,1);
    reg_z = order_m(reg_idx,1);
    for k = 1:length(nonreg_z)
        [~,nearest_idx] = min(reg_z-nonreg_z(k));
        order_m(order_m(:,1) == nonreg_z(k),idx) = 3; % Save translation method
        order_m(order_m(:,1) == nonreg_z(k),idx+1) = order_m(order_m(:,1) == reg_z(nearest_idx),idx+1);
        order_m(order_m(:,1) == nonreg_z(k),idx+2) = order_m(order_m(:,1) == reg_z(nearest_idx),idx+2);
    end
end

% Interpolate translations for images in the stack that were not tested
if align_stepsize >1
   z_interp = setdiff(ref_slices,z);
   interp_m = zeros(length(z_interp),size(order_m,2));
   interp_m(:,1) = z_interp;
   order_m = vertcat(order_m,interp_m);
   for j = 2:length(markers)
       % Index within order matrix
       idx = 2+length(markers)+4*(j-2);
       order_m(length(z)+1:end,idx) = 4; % Save translation method
       if length(z) == 1
           order_m(length(z)+1:end,idx+1) = order_m(1:length(z),idx+1);
           order_m(length(z)+1:end,idx+2) = order_m(1:length(z),idx+2);
       else
           order_m(length(z)+1:end,idx+1) = interp1(z,order_m(1:length(z),idx+1),z_interp,'linear','extrap');
           order_m(length(z)+1:end,idx+2) = interp1(z,order_m(1:length(z),idx+2),z_interp,'linear','extrap');
       end
   end
   [~,sort_idx] = sort(order_m(:,1));
    order_m = order_m(sort_idx,:);
end

% Save translations
% Remove combined channel column
order_m(:,length(markers)+1) = [];

% Make variable names
vars = cell(1,size(order_m,2));
vars{1} = 'Reference_Z';
for j = 2:length(markers)
   idx = length(markers) + 1 + 4*(j-2);
   c = char(markers(j));
   vars{j} =  strcat('Pixels_',c);
   vars{idx} = strcat('TransformType_',c);
   vars{idx+1} = strcat('X_Shift_',c);
   vars{idx+2} = strcat('Y_Shift_',c);
   vars{idx+3} = strcat('CC_',c);
end

% Convert to table
coreg_table = array2table(order_m,'VariableNames',vars);

% Replace transform type index with specified registration procedure
tvars = {'Not_Aligned','Registered','PC_Outlier','Low_Signal','Interpolated'}';
idxs = 1 + length(markers) + 4*((2:length(markers))-2);
order_m(:,idxs) = order_m(:,idxs)+1;
order_m(:,idxs(any(order_m(:,idxs) == 1,1))) = 1;

% Add transform type to coreg_table
t_table = cell2table(tvars(order_m(:,idxs)),'VariableNames',coreg_table(:,idxs).Properties.VariableNames);
coreg_table(:,idxs) = [];
coreg_table = [coreg_table t_table];

% Add image filenames
empty_files = repmat({''},height(coreg_table),1);
for i = fliplr(1:length(markers))
   file_table = table(empty_files,'VariableNames',{char(sprintf("file_%d",i))});
   coreg_table = cat(2,file_table,coreg_table);
   file_paths = path_full(path_full.markers == markers(i),:);
   file_paths = file_paths(file_paths.z >= min_z & file_paths.z <= max_z,:);
   coreg_table{ismember(coreg_table.Reference_Z,file_paths.z),1} = file_paths.file;
end

% Write images
if isequal(save_images,"true")
    write_aligned_images(config,coreg_table,res_equal,res_adj,row,col)
end

end


function write_aligned_images(config,coreg_table,res_equal,res_adj,row,col)

% Write image
fprintf("%s\t Writing aligned images \n", datetime('now'));

markers = config.markers;
save_dir = fullfile(config.output_directory,'aligned');

% Create directory to store images
if ~isfolder(save_dir)
    mkdir(save_dir);
end
    
% Get table index for translations for each marker
t_idx = zeros(1,length(markers));
for i = 2:length(markers)
    if i>1 && any(~ismember(config.align_channels,i))
        continue
    end
    t_idx(i) = find(contains(coreg_table.Properties.VariableNames,'X') &...
        contains(coreg_table.Properties.VariableNames,markers(i)));
end
    
% Subset images to save
if ~isempty(config.align_slices)
    save_z = config.align_slices{:};
    save_z = save_z(save_z<=height(coreg_table));
else
    save_z = 1:height(coreg_table);
end

idx = find(cellfun(@(s) ~isempty(s),coreg_table.file_1(save_z)),1);
ref_img = imread(coreg_table.file_1{idx});
    
% For each image in table
for i = save_z
    path_sub = coreg_table(coreg_table.Reference_Z == i,:);    
    for j = 1:length(markers)
        if j>1 && ~any(ismember(config.align_channels,j))
            continue
        end
        
        filepath = string(table2cell(coreg_table(i,j)));
        if filepath == ""
            mov_img = zeros(size(ref_img),'uint16');
        else
            mov_img = imread(filepath);

            % Apply intensity adjustments
            if isequal(config.adjust_intensity,"true")
               if isequal(config.adj_params.(markers(j)).adjust_tile_shading,'basic')
                   mov_img = apply_intensity_adjustment(mov_img,...
                       'flatfield', config.adj_params.(markers(j)).flatfield,...
                       'darkfield', config.adj_params.(markers(j)).darkfield);
               elseif isequal(config.adj_params.(markers(j)).adjust_tile_shading,'manual')
                  mov_img = apply_intensity_adjustment(mov_img,...
                      'y_adj',config.adj_params.(markers(j)).y_adj);
               end
            end
        end

        % Apply translations to non-reference image
        if j > 1
            % Adjust resample resolution if necessary
            if ~res_equal(j-1)
                mov_img = imresize(mov_img,res_adj{j-1},'bicubic');
            end

            % Crop or pad to reference image
            if any(size(ref_img) ~= size(mov_img))
                mov_img = crop_to_ref(ref_img, mov_img);
            end

            % Translate
            mov_img = imtranslate(mov_img,[path_sub{1,t_idx(j)} path_sub{1,t_idx(j)+1}]);
        end

        % Write aligned images
        img_name = sprintf('%s_%s_C%d_%s_0%d_0%d_aligned.tif',config.sample_id,num2str(coreg_table.Reference_Z(i),'%04.f'),j,markers(j),row,col);
        img_path = fullfile(save_dir,img_name);
        imwrite(uint16(mov_img),img_path)
    end
end


end
