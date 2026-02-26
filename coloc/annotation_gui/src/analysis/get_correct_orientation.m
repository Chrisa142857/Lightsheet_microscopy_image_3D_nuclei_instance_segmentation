function [img, perm] = get_correct_orientation(img, config, hemisphere, orientation)
% Orient a resampled to match the correct orientation. This is typically
% done on the atlas image. Outputs transformed image and permutation key.
% Default orientation is left hemisphere with imaging from the lateral side

if nargin == 2
    hemisphere = config.hemisphere;
    orientation = config.orientation;
end    

% Transform image based on which hemisphere is being imaged
if isequal(hemisphere, "right")
    perm(1) = 1;
    img = flip(img,1);
elseif isequal(hemisphere,"whole")
    perm(1) = 2;
    atlas_img2 = flip(img,3);
    img = cat(3,img,atlas_img2);
    if isequal(orientation,"lateral")
        warning('%s\t Using lateral orientation on whole brain image\n',datetime('now'));
    end
end

% Transform image based on imaging direction 
if isequal(orientation,"dorsal")
    perm(2) = 1;
    img = permute(img,[1,3,2]);
    img = flip(img,3);
elseif isequal(orientation,"ventral")
    perm(2) = 2;
    img = permute(img,[1,3,2]);
end

end