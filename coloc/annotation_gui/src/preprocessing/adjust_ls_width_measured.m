function y_adj = adjust_ls_width_measured(single_sheet, tempI, ls_width,...
    res, laser_y_displacement)
%--------------------------------------------------------------------------
% Adjust for laser width using measured intensity profiles specifically for
% the LaVision Ultramicroscope II. Measurements are based on intensity
% profiles from fluorescent dyes across multiple magngifications. Note this
% function requires Curve Fitting Toolbox.
%--------------------------------------------------------------------------
% Usage:
% y_adj = adjust_ls_width_measured(single_sheet, tempI, ls_width, res,
%                                  laser_y_displacement)
%
%--------------------------------------------------------------------------
% Inputs:
% single_sheet: ('true','false') Whether to use profile from single or 
% multiple light-sheet(s).
% 
% tempI: Example image for calculating image dimensions.
%
% ls_width: Light-sheet width setting as percentage.
%
% res: 1x3 double for image resolution as (um/voxel).
%
% laser_y_displacement: A value between -0.5 and 0.5 specifying a known 
% shift in laser position along the y axis. (default: 0)
%
%--------------------------------------------------------------------------
% Outputs:
% y_adj: Vector containing adjusted image intensity profile.
%
%--------------------------------------------------------------------------

% Set laser displacement
if nargin<5
    laser_y_displacement = 0;
end

fprintf('%s\t Adjusting For Laser Width \n',datetime('now'));    

if isequal(single_sheet,'true')
    % Measured intensities for single sheet acquisition 
    I_int = [436,18.5,4.52,2.58,1.89; 204,8.54,3.11,2.06,1.58;...
        104,4.50,2.21,1.56,1.34; 49.4,2.81,1.62,1.29,1.17;...
        8.63,1.93,1.31,1.14,1.07; 3.44,1.56,1.20,1.11,1.08;...
        2.19,1.31,1.11,1.06,1.05; 1.54,1.17,1.07,1.05,1.05;...
        1.27,1.08,1.04,1.04,1.04; 1.14,1.05,1.03,1.03,1.03;...
        1.07,1.03,1.02,1.02,1.02];
    stdev = [1251.8, 2113.8, 3110.0, 3935.5, 4522.3];
else
    % Measured intensities for 3 sheet acquistion
    I_int = [311,16.6,4.25,1.66,1.31; 171,8.19,2.95,1.41,1.20;...
        91.6,4.29,2.13,1.20,1.11; 44.8,2.79,1.57,1.08,1.04;...
        8.58,1.91,1.27,1.02,1.02; 3.26,1.52,1.17,1.02,1.02;...
        2.08,1.28,1.08,1.01,1.01; 1.45,1.14,1.04,1.01,1.01;...
        1.21,1.07,1.02,1.02,1.02; 1.12,1.04,1.02,1.02,1.02;...
        1.06,1.01,1.01,1.01,1.01];
    stdev = [1299.4, 2150.3, 3198.3, 4522.3, 4522.3];

end

% Pixel scales for acquired magnifications (um/pix)
I_pix = [4.16, 3.69, 2.98, 2.4, 1.86, 1.51, 1.21, 0.94, 0.75, 0.60, 0.49];

% Size of acuqisition window height
I_width = [20, 40, 60, 80, 100];
I_y = 1280.*I_pix;

stdev = zeros(1,5);
v = 1:round(I_pix(1)*1280);
for i = 1:size(I_int,2)
    % Take inverse intensity
    I_current = 1./I_int(:,i);
    % Fit a gaussian curve
    f=fit(I_y',I_current,'gauss1');
    g = feval(f,v);
    % Find FWHM
    [~,gg] = min(abs(g-0.5));
    % Calculate approximate std
    stdev(i) = 2*gg/2.355;
end

% Evaluate stdev at given width and scale
I_std = polyval(polyfit(I_width,stdev,1),ls_width)/res(1);

% Generate gaussian curve based on stdev
v2 = 1:10000;
I_fit = (1/(I_std*(2*pi)^0.5))*exp(-(v2).^2/(2*I_std^2));
w = 2-(I_fit./I_fit(1));

% Determine weights based on image size
height = size(tempI,1);

% Adjust center towards user-provided y_displacement
right = round(height*laser_y_displacement+height/2);
left = height-right;

% Combine to form intensity adjustment along y-axis 
gg_right = w(1:right);
gg_left = w(1:left);
y_adj = horzcat(fliplr(gg_left),gg_right);

% Normalize 
y_adj = y_adj/mean(y_adj);

end
