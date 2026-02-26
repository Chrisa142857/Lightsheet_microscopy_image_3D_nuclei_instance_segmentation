function [lowerThresh, upperThresh, signalThresh] = measure_thresholds(stack,defaults)
%--------------------------------------------------------------------------
% Get only intensity thresholds without additional adjustments. 
%--------------------------------------------------------------------------
%
% Inputs:
% stack - table of images to measure. Should contain only 1 channel.
%
% defaults - structure containing settings for image intensity measurement.
% Should contain lower/upper percentiles + image sampling proportion.
%
% Outputs:
% lowerThresh, upperThresh, signalThresh - unit-normalized lower and upper 
% thresholds as well as a rough global threshold for image signal.
%--------------------------------------------------------------------------

% Load defaults
low_prct = defaults.low_prct;                   % Low percentile for sampling background pixels
high_prct = defaults.high_prct;                 % High percentile for sampling bright pixels
image_sampling = defaults.image_sampling*0.2;   % Fraction of all images to sample    

% Count number of images and measure image dimensions
x_tiles = length(unique(stack.x));
y_tiles = length(unique(stack.y));
n_channels = length(unique(stack.markers));
nb_slices = length(unique(stack.z));

% Get image positions
s = round(image_sampling*nb_slices);
img_range = round(linspace(min(stack.z),max(stack.z),s));
if isempty(img_range)
    img_range = round(nb_slices/2);
end 

% Read image size and get padding
tempI = imread(stack.file{1});
[nrows, ncols] = size(tempI);
if isfield(defaults,'pads')
    pads = defaults.pads;
else
    pads = 0;
end
range_v = [round(pads*nrows)+1, nrows - round(pads*nrows)];
range_h = [round(pads*ncols)+1, ncols - round(pads*ncols)];
    
% Initialize measurement vectors
p_low = zeros(1,length(img_range));
p_high = zeros(1,length(img_range));
p_mad = zeros(1,length(img_range));

sub_low = zeros(1,x_tiles*y_tiles*n_channels); 
sub_high = zeros(1,x_tiles*y_tiles*n_channels);
sub_mad = zeros(1,x_tiles*y_tiles*n_channels);

% Take measurements
fprintf("%s\t Measuring thresholds... \n",datetime('now'))
for i = 1:length(img_range)
    % Read image regions where tiles should overlap
    files = stack(stack.z == img_range(i),:).file;
    
    for j = 1:length(files)
        img = single(imread(files{j},'PixelRegion',{range_v, range_h}));
        
        % Remove any zero values from shifting image
        img= img(img>1);
        
        if isempty(img)
            continue
        end
        
        % Measure percentiles of tiles
        sub_low(j) = prctile(img,low_prct,'all');
        sub_high(j) = prctile(img,high_prct,'all');
        sub_mad(j) = mad(img,1,'all');
    end 

    % Measure percentiles of all tiles
    p_low(i) = median(sub_low);
    p_high(i) = max(sub_high);
    p_mad(i) = mean(sub_mad);
end

% Remove 0 entries
p_low = p_low(p_low>0);
p_mad = p_mad(p_mad>0);

% Calculate thresholds
lowerThresh = median(p_low)/65535;
upperThresh = max(p_high)/65535;
signalThresh = lowerThresh + mean(p_mad)/65535;

fprintf('%s\t Measured lower, signal, & upper thresholds:\t %.1f\t %.1f\t %.1f\t\n',...
    datetime('now'),round(lowerThresh*65535),round(signalThresh*65535),round(upperThresh*65535));  

end
