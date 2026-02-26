function volumes = measure_structure_volumes(input)
% Take a registered annotation volume and calculate structure voxel
% volumes. Input is a NM_analyze config structure

% Check if string
if isstring(input)
    load(input,'I_mask')
else
    load(fullfile(input.output_directory,'variables',strcat(input.sample_id,'_mask.mat')),'I_mask')
end

% Calculate volume per voxel in mm^3
resolution = repmat(25,1,3);
mm = prod(resolution)/(1E3^3);

% Measure number of voxels for each structure
total_volume = sum(I_mask(:)>0)*mm;

% Read structure indexes from mask and count voxels
indexes = unique(I_mask(:));
sums = histcounts(I_mask(:),'BinLimits',[1,max(indexes)],'BinMethod','integers');
volumes = sums*mm';
fprintf('%s\t Total structure volume: %.2f mm^3\n',datetime('now'),total_volume)
volumes = [1:length(volumes);volumes]';
end