%This function applies a mask from a registered image at the isotropic
%resolution to a high resolution 2D image
function I_masked = apply_mask(I,res,I_mask,resample_res,z)

scaling_factors = res./resample_res;

z_scaled = round(z*scaling_factors(3));
z_scaled = max(1,z_scaled);

I_mask = I_mask(:,:,z_scaled);
I_mask = imresize(I_mask,size(I));

I_masked = I;
I_masked(I_mask ~= 1) = 0;
end