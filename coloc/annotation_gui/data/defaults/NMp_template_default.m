%% Template to run Tissue Clearing Processing Pipeline
% These are the key parameters
% Set flags to indicate how/whether to run process
adjust_intensity = "true";              % true, update, false; Whether to calculate and apply any of the following intensity adjustments. Intensity adjustment measurements should typically be performed on raw images
channel_alignment = "true";             % true, update, false; Channel alignment
stitch_images = "true";                  % true, update, false; 2D iterative stitching

use_processed_images = "false";         % false or name of sub-directory in output directory (i.e. aligned, stitched...); Load previously processed images in output directory as input images
ignore_markers = "Auto";                % completely ignore marker from processing steps. 
save_images = "true";                   % true or false; Save images during processing. Otherwise only parameters will be calculated and saved
save_samples = "true";                  % true, false; Save sample results for each major step

%% Intensity Adjustment Parameters
adjust_tile_shading = "basic";          % basic, manual, false; Can be 1xn_channels. Perform shading correction using BaSIC algorithm or using manual measurements from UMII microscope
adjust_tile_position = "true";          % true, false; Can be 1xn_channels. Normalize tile intensities by position using overlapping regions
darkfield_intensity = 101;              % 1xn_channels; Constant darkfield intensity value (i.e. average intensity of image with nothing present)
update_intensity_channels = [];         % integers; Update intensity adjustments only to certain channels 

% Parameters for manual shading corretion (i.e. adjust_tile_shading = "manual")
single_sheet = "true";                  % true, false; Whether a single sheet was used for acquisition
ls_width = 50;                          % 1xn_channels interger; Light sheet width setting for UltraMicroscope II as percentage
laser_y_displacement = 0;               % [-0.5,0.5]; Displacement of light-sheet along y axis. Value of 0.5 means light-sheet center is positioned at the top of the image

% Parameters for BaSIC shading_correction (i.e. adjust_tile_shading = "basic") 
sampling_frequency = 0.2;               % [0,1]; Fraction of images to read and sample from. Setting to 1 means use all images
shading_correction_tiles = [];          % integer vector; Subset tile positions for calculating shading correction (row major order). It's recommended that bright regions are avoid
shading_smoothness = 2;                 % numeric >= 1; Factor for adjusting smoothness of shading correction. Greater values lead to a smoother flatfield image
shading_intensity = 1;                  % numeric >= 1; Factor for adjusting the total effect of shading correction. Greater values lead to a smaller overall adjustment

%% Z Alignment Parameters
% Used for stitching and alignment by translation steps
update_z_adjustment = "false";          % true, false; Update z adjusment steps with new parameters. Otherwise pipeline will search for previously calculated parameters
z_positions = 0.01;                     % integer or numeric; Sampling positions along adjacent image stacks to determine z displacement. If <1, uses fraction of all images. Set to 0 for no adjustment, only if you're confident tiles are aligned along z dimension
z_window = 5;                           % integer; Search window for finding corresponding tiles (i.e. +/-n z positions)
z_initial = 0;                          % 1xn_channels-1 interger; Predicted initial z displacement between reference channel and secondary channel (i.e. 

%% Channel Alignment Parameters
align_method = "translation";           % elastix, translation; Channel alignment by rigid, 2D translation or non-rigid B-splines using elastix
align_tiles = [];                       % Option to align only certain stacks and not all stacks. Row-major order
align_channels = [];                    % Option to align only certain channels (set to >1)
align_slices = {};                      % Option to align only certain slice ranges. Set as cell array for non-continuous ranges (i.e. {1:100,200:300})

% Specific to translation method
align_stepsize = 10;                    % interger; Only for alignment by translation. Number of images sampled for determining translations. Images in between are interpolated
only_pc = "false";                      % true, false; Use only phase correlation for registration. This gives only a quick estimate for channel alignment. 

% Specific to elastix method
align_chunks = [];                      % Only for alignment by elastix. Option to align only certain chunks
elastix_params = "32_bins";             % 1xn_channels-1 string; Name of folders containing elastix registration parameters. Place in /supplementary_data/elastix_parameter_files/channel_alignment
pre_align = "false";                    % true, false; (Experimental) Option to pre-align using translation method prior to non-linear registration
max_chunk_size = 300;                   % integer; Chunk size for elastix alignment. Decreasing may improve precision but can give spurious results
chunk_pad = 30;                         % integer; Padding around chunks. Should be set to value greater than the maximum expected translation in z
mask_int_threshold = [];                % numeric; Mask intensity threshold for choosing signal pixels in elastix channel alignment. Leave empty to calculate automatically
resample_s = [3 3 1];                   % 1x3 integer. Amount of downsampling along each axis. Some downsampling, ideally close to isotropic resolution, is recommended
hist_match = 64;                        % 1xn_channels-1 interger; Match histogram bins to reference channel? If so, specify number of bins. Otherwise leave empty or set to 0. This can be useful for low contrast images

%% Stitching Parameters
% Parameters for running iterative 2D stiching
sift_refinement = "true";               % true, false; Refine stitching using SIFT algorithm (requires vl_fleat toolbox)
load_alignment_params = "true";         % true, false; Apply channel alignment translations during stitching
overlap = 0.10;                         % 0:1; overlap between tiles as fraction

stitch_sub_stack = [];                  % z positions; If only stitching a cetrain z range from all the images
stitch_sub_channel = [];                % channel index; If only stitching certain channels
stitch_start_slice = [];                % z index; Start stitching from specific position. Otherwise this will be optimized

blending_method = "sigmoid";            % sigmoid, linear, max
sd = 0.05;                              % 0:1; Recommended: ~0.05. Steepness of sigmoid-based blending. Larger values give more block-like blending
border_pad = 25;                        % integer >= 0; Crops borders during stitching. Increase if images shift significantly between channels to prevent zeros values from entering stitched image

%% Additional Post-processing Filters and Adjustments That May Be Useful But Are Not Required
% These all currently occur during stitching step after the completed image
% has been merged.
% Parameters for rescale_intensities
rescale_intensities = "false";          % true, false; Rescaling intensities and applying gamma
lowerThresh = [];                       % 1xn_channels numeric; Lower intensity for rescaling
signalThresh = [];                      % 1xn_channels numeric; Rough estimate for minimal intensity for features of interest
upperThresh = [];                       % 1xn_channels numeric; Upper intensity for rescaling
Gamma = [];                             % 1xn_channels numeric; Gamma intensity adjustment

% Background subtraction
subtract_background = "false";          % true, false. Subtrat background (similar to Fiji's rolling ball background subtraction)
nuc_radius = 13;                        % numeric >= 1; Max radius of cell nuclei along x/y in pixels. Required also for DoG filtering

% Difference-of-Gaussian filter
DoG_img = "false";                      % true,false; Apply difference of gaussian enhancement of blobs
DoG_minmax = [0.8,2];                   % 1x2 numeric; Min/max sigma values to take differene from.
DoG_factor = 1;                         % [0,1]; Factor controlling amount of adjustment to apply. Set to 1 for absolute DoG

% Smoothing filters
smooth_img = "false";                   % 1xn_channels, "gaussian", "median", "guided". Apply a smoothing filter
smooth_sigma = [];                      % 1xn_channels numeric; Size of smoothing kernel. For median and guided filters, it is the dimension of the kernel size

% Permute image. Sample orientation must be updated in downstream analysis
flip_axis = "none";                     % "none", "horizontal", "vertical", "both"; Flip image along horizontal or vertical axis
rotate_axis = 0;                        % 90 or -90; Rotate image
