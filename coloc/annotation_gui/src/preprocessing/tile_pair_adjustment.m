function img3 = tile_pair_adjustment(img1,img2,img3,tile_int,limit)

int_thresh = prctile(img1,tile_int,'all');

if int_thresh == 0
    return
end

idx = img1>int_thresh;
a2 = img1(idx)./img2(idx);

%a2(a2>1.2) = 1.2;
%a2(a2<0.8) = 0.8;
a2 = median(a2);

if isnan(a2)
    a2 = 1;
end

%img1 = sort(img1(idx),'descend');
%img2 = sort(img2(idx),'descend');

%s_idx = round(length(img1)*tile_int/100);
%a2 = mean(img1(s_idx)./img2(s_idx));

%img3 = img3*a2;

end