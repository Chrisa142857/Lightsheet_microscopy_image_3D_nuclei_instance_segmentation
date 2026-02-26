%% Template to run Tissue Clearing Analysis Pipeline
% These are the key parameters
% Set flags to indicate how/whether to run process
resample_images = "true";           % true, update, false; Perform image resampling 
register_images = "true";           % true, update, false; Register image to reference atlas.
count_nuclei = "true";              % true, update, false; Count cell nuclei or other blob objects.
classify_cells = "true";            % true, update, false; Classify cell-types for detected nuclei centroids.

use_processed_images = "stitched";  % false or name of sub-directory in output directory (i.e. aligned, stitched...); Direct pipeline to load previously processed images in output directory

%% Resampling and Annotation Parameters
% Resampling Parameters
resample_resolution = 25;           % Isotropic resample resolution. This is also the resolution at which registration is performed
resample_channels = [];             % Resample specific channels. If empty, only registration channels will be resampled

% For annotations
use_annotation_mask = "true";       % true, false; Use annotation mask for cell counting
annotation_mapping = "atlas";       % atlas, image; Specify whether annotation file is mapped to the atlas or light-sheet image
annotation_file = "ccfv3.mat";      % File for storing structure annotation data. 
                                    % Specify .mat file in /data/annotation_data if annotation_mapping is to the atlas. To generate a new .mat file with custom annotations, see 'help munge_atlas'
                                    % Specify .nii file if annotation_mapping is to the image. Annotations here are already aligned to the image. Specify the full path for each sample in NM_samples.

use_structures = "structure_template.csv";      % Specify csv file in /annotations detailing which structures to analyze
                                                % Alternatively, specify a numeric array structure indexes 
annotation_resolution = 25;                     % Isotropic resolution of the annotation file. Only needed when mapping is to the image.

%% Registration Parameters
registration_direction = "atlas_to_image";      % atlas_to_image, image_to_atlas; Direction to perform registration
registration_parameters = "default";            % default, points, or name of folder containing elastix registration parameters in /data/elastix_parameter_files/atlas_registration
registration_channels = 1;                      % integer; Which light-sheet channels to register. Can select more than 1
registration_prealignment = "image";            % image. Pre-align multiple light-sheet images by rigid transformation prior to registration

atlas_file = "ara_nissl_25.nii";                % ara_nissl_25.nii and/or average_template_25.nii and/or a specific atlas .nii file in /data/atlas
points_file = [];                               % Name of points file to guide registration
save_registered_images = "true";                % Whether to save registered images
mask_cerebellum_olfactory = "true";             % Remove olfactory bulbs and cerebellum from atlas ROI

%% Nuclei Detection
count_method = "3dunet";                % 3dunet, hessian. 
min_intensity = 200;                    % Minimum intensity for cell nuclei

% 3D-Unet specific
model_file = "075_121_model.h5";        % Model file name located in /analysis/3dunet/nuclei/models
gpu = '0';                              % Cuda visible device index. 

% Hessian blob detector
chunk_size = [1000,1000,32];            % Max chunk size for running hessian 
chunk_overlap = [25,25,8];              % Chunk overlap
average_nuc_diameter = 11;              % Average nucleus diameter in pixels

%% Cell-Type Classification
classify_method = "threshold";              % threhsold, svm; Cell-type classification method
classify_channels = [];                     % which channels to use for classification
contains_nuclear_channel = "true";          % "true","false"; If set to "true", treat channel 1 as nuclear label that isn't classified 
remeasure_centroids = "false";              % true, false; Update channel intensities measurements and save into centroids sheet

% Threshold classification
intensity_thresholds = [];                  % Set raw intensity values for thesholding. Leave empty to use expression values
intensity_expression = "1*mode + 3*mad";    % Threshold expression
z_normalization = "true";                   % Apply z normalization

% Support-Vector Machine (svm) classification
load_patches = "true";                          % Load previous centroid patches
load_groups = [];                               % Merge patch annotation from different groups for training
keep_classes = [];                              % Which classes to keep after prediction. Other labeled classes will be discarded
patch_size = [50,6];                            % 1x2 integer. [Patch size for viewing, patch size read by classifier]
n_patches = 1000;                               % integer. Number of patches to generate
min_class_thresh = 0.5;                         % numeric <1. Remove cells dim in every channel from classifier. (i.e. 0.5 removes cells below 50th percentile for all channels)

%% Additional carry-over threshold values from NM_process
% Can leave these empty if images went through processing steps
lowerThresh = [];                       % 1xn_channels numeric; Lower intensity for rescaling
signalThresh = [];                      % 1xn_channels numeric; Rough estimate for minimal intensity for features of interest
upperThresh = [];                       % 1xn_channels numeric; Upper intensity for rescaling
