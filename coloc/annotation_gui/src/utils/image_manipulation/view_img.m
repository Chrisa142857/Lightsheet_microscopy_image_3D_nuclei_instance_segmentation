function view_img(input,index,adjust)
%--------------------------------------------------------------------------
% Wrapper function to view images of various types.
%--------------------------------------------------------------------------
% Usage: 
% view_img(filepath,index,adjust);
%
%--------------------------------------------------------------------------
% Inputs:
% filepath: (table, string, or numeric) Pass path_table to read and view a 
% single image entry. Pass a string to file location to read and view
% image. Pass a matrix to view slice.
%
% index: (up to 1x4 numeric) Channel, slice, row, column position to gather
% from table to view. 
%
% adjust: (logical) Rescale image intensities. (Default: true)
%
%--------------------------------------------------------------------------

if istable(input) && nargin<2
    index = ones(1,4);
elseif istable(input) && length(index)<4
    index = [index, ones(1,4-length(index))];
else
    index = [];
end

if nargin<3
    adjust = true;
end

if ~isempty(index) && istable(input)
    input = input(input.channel_num == index(1) &...
        input.z == index(2),:);
    if height(input) > 1
        input = input(input.y == index(3) &...
            input.x == index(4),:);
    end
    if ~isempty(input)
        input = input.file{1};
    else
        error("Channel/position index is out of range.")
    end
end

% Read image if input is string
if isstring(input) || ischar(input)
    [~,~,ext] =  fileparts(input);
    switch ext
        case {'.tif','.tiff'}
            if imfinfo(input).FileSize >1E7
                img = loadtiff(input);
            else
                img = imread(input);
            end
        case {'.nii','.mhd'}
            img = niftiread(input);
            img = img(:,:,index(2));
        case {'.nrrd','.nhdr'}
            out = nhdr_nrrd_read(input,true);        
            img = out.data(:,:,index(2));
        otherwise
            error("Unrecognized file type.")
    end
else
    img = input;
end

if adjust
    figure; imshow(imadjust(img))
else
    figure; imshow(img)
end


end