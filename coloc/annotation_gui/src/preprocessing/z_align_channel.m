function [z_displacement,ave_score] = z_align_channel(config,path_mov,path_ref,channel_idx)
%--------------------------------------------------------------------------
% Align .tif series from 2 channels along z dimension.
%--------------------------------------------------------------------------

% Set number of peaks and subixel value
peaks = 3;              % Phase correlation peaks to test
usfac = 1;              % Precision of phase correlation
max_shift = 0.2;        % Max shift as fraction of longest image dimension
min_signal = 0.05;       % Minimum fraction of signal pixels in reference for registering

% Unpack variables
z_positions = config.z_positions;
z_window = config.z_window;
refThresh = config.signalThresh(1);
signalThresh = config.signalThresh(channel_idx);
z_initial = config.z_initial(channel_idx);

% Adjust for resolution
res_adj = config.resolution{1}./config.resolution{channel_idx};
res_equal = all(res_adj == 1);

% Get shift threshold as pixels
tempI = imread(path_ref.file{1});
npix = numel(tempI);
shift_threshold = max(size(tempI))*max_shift;

% Check resolutions
if ~res_equal
    res_adj = size(tempI).*res_adj(1:2);
    shift_threshold = max(res_adj)*max_shift;
end

% Adjust signalThresh to 16 bit
if signalThresh<1
    signalThresh = signalThresh*65535;
end
if refThresh<1
    refThresh = refThresh*65535;
end

% Read number of reference/moving images
nb_mov = height(path_mov);    
nb_ref = height(path_ref);

% Adjust z_positions if less than 1
if z_positions<1
    z_positions = ceil(nb_ref*z_positions);
end

% Adjust z_positions if specified number is too high
z_positions = min(z_positions,nb_mov-z_window*2);

% Pick reference images in range
if z_positions == 1
    z = round(nb_ref/2);
else
    z = round(linspace(z_window+1,nb_ref-z_window,z_positions));
end
path_ref = path_ref(z,:);

% Initial matrix of cross correlation values and perform registration using
% cross correlation
cc = zeros(length(z),z_window*2+1);
max_signal = zeros(1,length(z));
for i = 1:length(z)
    % Read reference image and specify z range to look based on user-defined
    % window
    z_range = z(i)-z_window+z_initial:1:z(i)+z_window+z_initial;
    ref_img = read_img(path_ref.file{i});
    
    % If low intensity in this image, continue to next one
    refSignal = sum(ref_img>refThresh,'all')/npix;
    if refSignal < min_signal
        continue
    end

    % Resample the reference image if the resolution is different
    % Note: we're resizing reference image instead of moving image for
    % speed and because moving image is usually at lower resolution
    if ~res_equal
        ref_img = imresize(ref_img, res_adj);
    end

    % Compare moving images in range
    for j = 1:length(z_range)
        % Continue if z_position is not in range
        if z_range(j) <1 || z_range(j)>nb_mov
            continue
        end
        
        % Read moving image
        mov_img = read_img(path_mov.file{z_range(j)});

        % Calculate number of bright pixels
        signal = numel(mov_img(mov_img>signalThresh))/numel(mov_img);
        max_signal(i) = max(max_signal(i),signal);
        
        % If no bright pixels, continue to next image
        if signal<min_signal
            continue
        end
        
        % Do simple crop in case image sizes don't match up
        if any(size(ref_img) ~= size(mov_img))
            mov_img = crop_to_ref(ref_img,mov_img);
        end    
        
        %Calculate phase correlation
        [~,~,~,type,cc(i,j)] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold);        

        %If pc didn't work, set cc to 0
        if type == 0
            cc(i,j) = 0;
        end

        %Check results
        %[pc_img,ref_img2,~,type,cc(a,b)] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold);
        %if i == 1
            %figure
            %imshowpair(ref_img2*30,uint16(pc_img)*25)
        %end
    end
end

% Generate score by multiplying intensity value by cross-correlation. This
% will lower effect caused by noise in images with no features
[val, pos] = max(cc'.*max_signal);
pos_idx = unique(pos);

% Sum scores for all unqiue z_positions
score = zeros(1,length(pos_idx));
for i = 1:length(pos_idx)
    score(i) = sum(val(pos==pos_idx(i)));    
end

% Max score determines final z displacement
[~,k] = max(score);

% Display cross-correlation matrix
disp(mean(cc,1))

% Adjust final z displacement based on initial z position
z_displacement = pos_idx(k)-(z_window+1)+z_initial;
ave_score = mean(val);

end
