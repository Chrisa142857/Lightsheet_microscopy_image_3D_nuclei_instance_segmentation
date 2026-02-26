function I = postprocess_image(config, I, marker_idx)
%--------------------------------------------------------------------------
% Postprocess an image by filtering, subtracting background, or enhancing
% blobs as specified in the config structure.
%--------------------------------------------------------------------------

% Check for length of marker_idx
n_markers = length(config.markers);
if marker_idx > length(config.rescale_intensities)
    config.rescale_intensities = repmat(config.rescale_intensities,1,n_markers);
end
if marker_idx > length(config.subtract_background)
    config.subtract_background = repmat(config.subtract_background,1,n_markers);
end
if marker_idx > length(config.DoG_img)
    config.DoG_img = repmat(config.DoG_img,1,n_markers);
end
if marker_idx > length(config.smooth_img)
    config.smooth_img = repmat(config.smooth_img,1,n_markers);
end

% Rescale intensities
if isequal(config.rescale_intensities(marker_idx),"true")
    marker = config.markers(marker_idx);
    l_thresh = config.adj_params.(marker).lowerThresh;
    u_thresh = config.adj_params.(marker).upperThresh;
    gamma = config.adj_params.(marker).Gamma;
    I = apply_intensity_adjustment(I,'l_thresh',l_thresh,'u_thresh',u_thresh,'gamma',gamma);
end

% Apply background subtraction
if isequal(config.subtract_background(marker_idx),"true")
    I = smooth_background_subtraction(I, false, config.nuc_radius);
end

% Apply Difference-of-Gaussian filter
if isequal(config.DoG_img(marker_idx),"true")
    try
        I = dog_adjust(I,config.nuc_radius,config.DoG_minmax,config.DoG_factor(marker_idx));
    catch
        I = dog_adjust(I,config.nuc_radius,config.DoG_minmax,config.DoG_factor(1));
    end
end

% Apply smoothing filter
if ~isequal(config.smooth_img(marker_idx),"false")
    try
        I = smooth_image(I, config.smooth_img(marker_idx),config.smooth_sigma(marker_idx));
    catch
        I = smooth_image(I, config.smooth_img(marker_idx),config.smooth_sigma(1));
    end
end

% Apply permutations
if isfield(config,"flip_axis")
    if isequal(config.flip_axis,"horizontal")
        I = flip(I,1);
    elseif isequal(config.flip_axis,"vertical")
        I = flip(I,2);
    elseif isequal(config.flip_axis,"both")
        I = flip(I,1);
        I = flip(I,2);
    end
end
if isfield(config,"rotate_axis")
    if config.rotate_axis ~= 0
        I = imrotate(I, config.rotate_axis);
    end
end

end