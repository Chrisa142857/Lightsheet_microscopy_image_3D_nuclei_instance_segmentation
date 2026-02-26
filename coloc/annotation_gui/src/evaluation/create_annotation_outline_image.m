function I = create_annotation_outline_image(I_mask,stack,z_pos,slice)

labels = unique(I_mask(:));
n_labels = length(labels);

%I_mask = imresize3(I_mask,[752,588,416],'Method','nearest');

[~,~,nslices] = size(I_mask);
I_outline = uint8(zeros(size(I_mask)));

resized_slice = round((slice/z_pos)*nslices);

img = I_mask(:,:,resized_slice);

for i = 1:height(stack)
    I(:,:,i) = imread(stack.file{i});
end

[nrows,ncols,~] = size(I);
img = I_mask(:,:,resized_slice);
img = imresize(img,[nrows,ncols],'Method','nearest');

bound = zeros([nrows,ncols]);
    
labels2 = unique(img(:));
n_labels2 = length(labels2);
    
for j = 2:n_labels2
    sub = img == labels2(j);
    %sub = imerode(sub,[0 1 0;1 1 1;0 1 0]);
    %sub = bwperim(sub,4);
    bound = bound + sub*labels2(j);
end
    
I(:,:,4) = bound;


end