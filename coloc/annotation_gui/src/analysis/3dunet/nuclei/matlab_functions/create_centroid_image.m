function create_centroid_image(cen_predict,img,save_name)
n_predict = size(cen_predict,1);

cen_img = uint16(zeros(size(img)));
for i = 1:n_predict
    pos = cen_predict(i,:);
    cen_img(pos(1),pos(2),pos(3)) = 1;
end

se = [0 1 0; 1 1 1; 0 1 0];
for i = 1:size(cen_img,3)
   cen_img(:,:,i) = imdilate(cen_img(:,:,i),se);
end

img = uint16(img*3);
cen_overlay = imoverlay(img(:,:,1),cen_img(:,:,1),[1,0,0]);
imwrite(cen_overlay,save_name)
for i = 2:size(cen_img,3)
   cen_overlay = imoverlay(img(:,:,i),cen_img(:,:,i),[1,0,0]);
   imwrite(cen_overlay,save_name,'WriteMode','append'); 
end

end