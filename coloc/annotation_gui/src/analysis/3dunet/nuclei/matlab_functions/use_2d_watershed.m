function prediction = use_2d_watershed(prediction)

% Apply 2D watershed to the prediction
for j = 1:size(prediction,3)
    pslice = prediction(:,:,j);
    D = bwdist(~pslice);
    L = watershed(-D);
    L(~pslice) = 0;
    prediction(:,:,j) = L;
end

end