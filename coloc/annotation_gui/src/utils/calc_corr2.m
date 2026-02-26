function cc = calc_corr2(img1,img2)
%--------------------------------------------------------------------------
% Calculate 2D correlation coefficient while ignoring zeroed areas.
%--------------------------------------------------------------------------

idx = img1 == 0 | img2 == 0;
cc = corr2(img1(~idx),img2(~idx));

end