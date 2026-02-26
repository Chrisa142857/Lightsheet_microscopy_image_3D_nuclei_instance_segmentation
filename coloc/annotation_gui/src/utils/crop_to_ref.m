%This function takes 2x2D images and either crops or pads according to the
%dimensions of the reference image
function mov_img2 = crop_to_ref(ref_img,mov_img)

[nrows_ref, ncols_ref] = size(ref_img);
[nrows_mov, ncols_mov] = size(mov_img);

mov_img2 = zeros(size(ref_img));
center_ref = [round(nrows_ref/2),round(ncols_ref/2)];
crop_flag = 0;

%Vertical pixels to crop
top = 1;
if nrows_mov > nrows_ref
    crop_flag = 1;
    top = round((nrows_mov-nrows_ref)/2);
end

%Horizontal pixels to crop
left = 1;
if ncols_mov > ncols_ref
    crop_flag = 1;
    left = round((ncols_mov-ncols_ref)/2);
end

%Crop
if crop_flag == 1
   mov_img = imcrop(mov_img,[left top ncols_ref-1 nrows_ref-1]);
end

[nrows_mov, ncols_mov] = size(mov_img);

%Vertical pixels to pad
left = 1;
if ncols_mov < ncols_ref
    left = round((ncols_ref-ncols_mov)/2);
end

%Horizontal pixels to pad
top = 1;
if nrows_mov < nrows_ref
   top = round((nrows_ref-nrows_mov)/2);
end

%Pad
mov_img2(top:(nrows_mov+top-1),left:(ncols_mov+left-1)) = mov_img;
end