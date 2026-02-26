function ftable = measure_patch_features(img, s, crop, sample)
%--------------------------------------------------------------------------
% Measure intensity/shape features in a 2D image patch. Returns table
% Intensity features: max/mean/middle intensity, st. dev., mid/corner voxels
% Shape features: solidity, filled area, distance from center
%--------------------------------------------------------------------------

if crop
    mid = round(length(img)/2);
    img = img(mid-s:mid+s,mid-s:mid+s);
end

s1 = round(s/2);

bw = imbinarize(img);
img = single(img);
ftable = zeros(1,8);

% Intensity features
ftable(1) = max(img(:));
ftable(2) = mean(img(:));
ftable(3) = std(img(:));
p_corner = mean([img(1,1), img(1,end), img(end,1), img(end,end)]);
p_middle = img(s+1,s+1);
ftable(4) = p_middle;
ftable(5) = p_middle/p_corner;
ftable(6) = mean(bw(s+1-s1:s+1+s1,s+1-s1:s+1+s1),'all');
ftable(7) = sum(bw(:));
%ftable(7) = ftable(6)/sum(bwconvhull(bw,'objects'),'all');
ftable(8) = 0;

% Shape features
%rp = regionprops(bw,'Solidity','FilledArea');
%try
%    ftable(6) = rp.Solidity;
%    ftable(7) = rp.FilledArea;
%catch
%    ftable(6) = 0;
%    ftable(7) = 0;
%end

if nargin == 4
    col_names = ["Max","Mean","Std","Middle","MoC","Solidity","FilledArea","CenDist"];
    col_names = col_names + sprintf('_%s',sample);
    ftable = array2table(ftable,'VariableNames',col_names);
end

end