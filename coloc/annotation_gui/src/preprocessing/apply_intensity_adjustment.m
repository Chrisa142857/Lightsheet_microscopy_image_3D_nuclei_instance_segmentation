function I = apply_intensity_adjustment(I,varargin)
%--------------------------------------------------------------------------
% Apply individual or a series of intensity adjustments. See parser inputs
% for details.
%--------------------------------------------------------------------------

% Default darkfield value
dark_val = 101;

% Parse inputs
p = inputParser;
addOptional(p, 'params', [], @isstruct) % Adjustment params structure
addOptional(p, 't_adj', false, @isnumeric) % Tile intensity adjustment value
addOptional(p, 'y_adj', false, @isnumeric) % LS width adjustment vector
addOptional(p, 'l_thresh', 0, @isnumeric) % Lower threshold rescale value
addOptional(p, 'u_thresh', 1, @isnumeric) % Upper threshold rescale value
addOptional(p, 's_thresh', 0, @isnumeric) % Signal threshold value
addOptional(p, 'Gamma', 1, @isnumeric) % Gamma adjustment value
addOptional(p, 'flatfield', false, @isnumeric) % Flatfield matrix
addOptional(p, 'darkfield', false, @isnumeric) % Darkfield matrix
addOptional(p, 'r', 1, @isnumeric) % Row index
addOptional(p, 'c', 1, @isnumeric) % Column index

parse(p,varargin{:})

% Get values
t_adj = p.Results.t_adj;
y_adj = p.Results.y_adj;    
l_thresh = p.Results.l_thresh;
u_thresh = p.Results.u_thresh;
s_thresh = p.Results.s_thresh;
Gamma = p.Results.Gamma;
flatfield = p.Results.flatfield;
darkfield = p.Results.darkfield;

% If adjustment parameter structure is sent in, use these values
adj_params = p.Results.params; % Adj params
r = p.Results.r; % Row
c = p.Results.c; % Column
if ~isempty(adj_params)
    if isequal(adj_params.adjust_tile_position,'true') && ~isempty(adj_params.t_adj)
        t_adj = adj_params.t_adj(r,c);
        s_thresh = adj_params.signalThresh;
    end
    if isequal(adj_params.adjust_tile_shading,'basic')
       flatfield = adj_params.flatfield;
       darkfield = adj_params.darkfield;
    elseif isequal(adj_params.adjust_tile_shading,'manual')
        y_adj = adj_params.y_adj;
        s_thresh = adj_params.lowerThresh;
    end
    if ~isempty(adj_params.darkfield_intensity)
        dak_val = adj_params.darkfield_intensity;
    end
end

% Check to see if single
img_class = class(I);
if ~isequal(img_class,'single')
    I = single(I);
end

% Adjust for tile intensity differences
if isnumeric(t_adj) && s_thresh > 0
    Imin = s_thresh*65535;
    I_dark = I-Imin;
    I = (I_dark*t_adj) + Imin;
elseif isnumeric(t_adj)
    Imin = dark_val;
    I_dark = I-Imin;
    I = (I_dark*t_adj) + Imin;
end

% Apply flatfield correction
if isnumeric(flatfield)
    if isnumeric(darkfield)
        I = (I-darkfield)./flatfield + darkfield;
    else
        I = I./flatfield;
    end
end

% Adjust for light-sheet width
if isnumeric(y_adj)
   % Crop laser width adjustment if necessary
    if length(y_adj) ~= size(I,1)
        y_ref = 1:size(I,1);
        y_adj = crop_to_ref(y_ref,y_adj);
    end
    if s_thresh > 0
        a = (s_thresh*65535)*(y_adj-1);
        I = I.*y_adj - a;
    else
        a = dark_val*(y_adj-1);
        I = I.*y_adj - a;
    end
end

% Rescale intensities
if u_thresh<1 || Gamma ~= 1
    I = imadjust(uint16(I), [l_thresh u_thresh], [0 1], Gamma);
end

% Recast to original class if necessary
if ~isequal(I, img_class)
    I = cast(I,img_class);
end

end
