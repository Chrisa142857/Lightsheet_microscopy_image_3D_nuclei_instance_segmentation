function apply_transform_to_resampled(mov_img_path,reg_params)

% Load atlas + moving image 
mov_img = niftiread(mov_img_path);
mov_img = imrotate(mov_img,90);
mov_img = flip(mov_img,1);

% Resize image
mov_img = imresize3(mov_img,0.4);

% Apply transform
reg_img = transformix(mov_img,reg_params,[1 1 1],[]);

% Write registered image?
fprintf('%s\t Saving registered image\n',datetime('now')); 

reg_img = uint16(reg_img);
reg_img = imadjustn(reg_img,stretchlim(uint16(mov_img(:))));
J = im2uint8(reg_img);

save_path = sprintf('%s_registered.tif',mov_img_path(1:end-4));
imwrite(J(:,:,1), save_path)
for i = 2:size(J,3)
   imwrite(J(:,:,i),save_path,'WriteMode','append'); 
end
end
