function I = smooth_image(I, type, sigma)
%--------------------------------------------------------------------------
% Smoothi image using one of the following filters: gaussian, median,
% guided.
%--------------------------------------------------------------------------

% Some defaults
guided_amount = 1E8;

switch type
    case 'false'
        return
    case 'gaussian'
        % 2D Gaussian smoothing
        I = imgaussfilt(I,sigma);
    case 'median'
        % Median filter
        I = medfilt2(I,[sigma sigma]);
    case 'guided'
        % Apply image guided filtering. This may preserve feature edges
        % slightly better than gaussian smoothing
        I = imguidedfilter(I,'DegreeofSmoothing',guided_amount,'NeighborhoodSize',[sigma sigma]);
    otherwise
        error("Unknown filter type")
end

end