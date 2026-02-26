function [pc_img,ref_img,tform,type,cc] = calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold)
%--------------------------------------------------------------------------
% calculate_phase_correlation Predict x,y translation by phase correlation.
% If no valid translations produced, this will also use MATLAB's imregcorr
% function
%
% Syntax:  [pc_img,ref_img,tform,type,cc] =
% calculate_phase_correlation(mov_img,ref_img,peaks,usfac,shift_threshold)
% 
% Inputs:
%   mov_img - (numeric) image to translate
%   ref_img - (numeric) reference image
%   peaks - (numeric, optional) number of phase correlation peaks to test.
%   Default is 3.
%   usfac - (numeric, optional) subpixel precision. Default is 1 (round to
%   interger). Value of 10 means round to nearest tenth.
%   shift_threshold - (numeric, optional) maximum valid translation in
%   either x or y. Default is Inf. 
%
% Output:
%   pc_img - (numeric) translated image
%   ref_img - (numeric) reference image
%   tform - (structure) resulting translation as MATLAB transformation 
%   object
%   type - 1,2,3. Numeric specifying which registration type was ultimately
%   used. 1 is phase correlation. 2 is imregcorr with no windowing. 3 is
%   imregcorr with windowing
%   cc - (numeric) cross correlation value
%--------------------------------------------------------------------------

% Set number of peaks as 3 unless specified
if nargin<3 || isempty(peaks)
    peaks = 3;
end

% Set precision as 1 pixel unless specified
if nargin<4 || isempty(usfac)
    usfac = 3;
end

% No max shift threshold if left unspecified
if nargin <5 || isnan(shift_threshold)
    shift_threshold = Inf;
end

warning('off','images:imregcorr:weakPeakCorrelation')

% Perform subpixel registration using phase correlation. 
% Output is y_translation;x_translation
output = dftregistration(ref_img,mov_img,peaks,usfac,shift_threshold);
type = 1;

if isempty(output)
% If pc using dft doesn't give anything useful, try MATLAB's imregcorr. This
% can sometimes give something using useful when SNR in one of the images is
% low. However only top peak is chosen
    tform = imregcorr(mov_img,ref_img,'translation','Window',false);
    output = [tform.T(6); tform.T(3)];
    type = 2; 
    % Remove if contains large shifts
    if abs(output(1))>shift_threshold || abs(output(2))>shift_threshold
        output = [];
    end
    % Try first without windowing. Then try with windowing
    if isempty(output)
        tform = imregcorr(mov_img,ref_img,'translation','Window',true);
        output = [tform.T(6); tform.T(3)];
        type = 3; 
        % Remove if contains large shifts
        if abs(output(1))>shift_threshold || abs(output(2))>shift_threshold
            output = [];
        end
    end
end

%If still no result, give empty transform
if isempty(output)
    pc_img = mov_img;
    tform = [];
    
    type = 0;
else
    %Translate image
    pc_img = imtranslate(mov_img,[output(2) output(1)]);

    %Save translation as affine2d object
    tform = affine2d([1 0 0; 0 1 0; output(2) output(1) 1]);
    
end

if nargout==5
    %Calculate cross-correlation
    cc = calc_corr2(ref_img,pc_img);
end

end
