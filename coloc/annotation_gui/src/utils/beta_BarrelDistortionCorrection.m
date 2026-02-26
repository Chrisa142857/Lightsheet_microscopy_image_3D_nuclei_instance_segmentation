%%
clear xTrue yTrue
clear xBad yBad
[rows, columns]= size(I);

 % Specify the distortion correction factor.
 % Positive to correct pincushion distortion, and bring coreners inwards.
 % Negative to correct barrel (fisheye) distortion, and push corners outward.
 df = -0.001; 
 xCenter = columns / 2;  % Assume optical axis in center of image.
 yCenter = rows / 2;
 samplingRate = 100;
 [xBad, yBad] = meshgrid([1 : samplingRate : columns], [1: samplingRate   :rows]); 
 plot(xBad, yBad, 'r+', 'MarkerSize', 5, 'LineWidth', 2);
 hold on;
 for r = 1 : size(xBad, 1)
  for c = 1 : size(xBad, 2)
    x = xBad(r, c);
    y = yBad(r, c);
    % Get the actual distance from the optic axis.
    rBad = sqrt((x - xCenter)^2 + (y - yCenter)^2);
    % Get the delta R that is moved.
    deltaRadius = df * rBad / (1 + df);
    rTrue = rBad - deltaRadius;
    angle = atan2((y - yCenter), (x - xCenter));
    xTrue(r, c) = xCenter + rTrue * cos(angle);
    yTrue(r, c) = yCenter + rTrue * sin(angle);
        D = interp2(double(I), x-xTrue, y-yTrue);
        % display the result
        %imshow(D, []);
  end
 end
plot(xTrue, yTrue, 'bo', 'MarkerSize', 5, 'LineWidth', 2);
legend({'+ = Distorted'; 'o = Corrected'});
hold off