function centroids = watershed_centroid(img)
% Generate centroid list using otsu+watershed
%img2 = adaptthresh(img);

img = imgaussfilt3(img,0.5);
bw = imbinarize(img);

D = bwdist(~bw);
D = -D;
L = watershed(D);
L(~bw) = 0;

L = bwareaopen(L,10);

rp = regionprops(L);
centroids = reshape([rp.Centroid],[3,length(rp)])';

centroids2 = [centroids(:,2), centroids(:,1), centroids(:,3)];
centroids = round(centroids2);
end