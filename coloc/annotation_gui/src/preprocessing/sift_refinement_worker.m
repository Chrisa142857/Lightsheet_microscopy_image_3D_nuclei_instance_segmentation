function tform = sift_refinement_worker(mov_img,ref_img)

% Defaults
min_distance = 25;

% Crop to overlapping region
if size(ref_img,1)>size(ref_img,2)
    % Horizontal
    idx = find(any(mov_img,1));
    mov_img = mov_img(:,min(idx):max(idx));
    ref_img = ref_img(:,min(idx):max(idx));
    ncols = size(mov_img,2);
    x = -floor(ncols/2):floor(ncols/2);
    weight = -x.^2+1;
    weight = (weight - min(weight))./(max(weight) - min(weight));
else
    % Vertical
    idx = find(any(mov_img,2));
    mov_img = mov_img(min(idx):max(idx),:);
    ref_img = ref_img(min(idx):max(idx),:);
    nrows = size(mov_img,1);
    x = -floor(nrows/2):floor(nrows/2);
    weight = (-x.^2+1)';
    weight = (weight - min(weight))./(max(weight) - min(weight));
end

% Detect SIFT features
[f1, d1] = vl_sift(ref_img,'PeakThresh',10,'EdgeThresh',2);
[f2, d2] = vl_sift(mov_img,'PeakThresh',10,'EdgeThresh',2);

% Match SIFT features
matches = vl_ubcmatch(d1, d2);

% Take x,y positions of matched points
x1 = f1(1:2,matches(1,:));
x2 = f2(1:2,matches(2,:));

% Calculate distance between matches
x3  = sqrt(sum((x1 - x2).^2));

% Remove point far away from each other
matches(:,abs(x3)>min_distance)=[];

% Display number of matches
numMatches = size(matches,2);

X1 = f1(1:2,matches(1,:)); X1(3,:) = 1;
X2 = f2(1:2,matches(2,:)); X2(3,:) = 1;

% Add non-linear blend weight to weigh matches in the middle of the images,
% (where the blending seams occur) more strongly
w = 1+weight.*((1-weight)/0.25);

if size(ref_img,1)>size(ref_img,2)
    w = w(ceil(X1(1,:)));
else
    w = w(ceil(X1(2,:)))';
end

%For plotting points
%imshow(imadjust(uint16(ref_img)))
%h1 = vl_plotframe(X1)';
%h2 = vl_plotframe(X2)';

%set(h1,'color','k');
%set(h2,'color','y');

if numMatches > 3
    % Instead of RANSAC, use just the average feature locations since we're
    % already removing outliers based on distance
    X3(1,:) = (X1(1,:)-X2(1,:)).*w;
    X3(2,:) = (X1(2,:)-X2(2,:)).*w;

    x = sum(X3(1,:))/sum(w);
    y = sum(X3(2,:))/sum(w);

    tform = affine2d([1 0 0; 0 1 0; x y 1]);
else
    %fprintf(strcat(char(datetime('now')),'\t Not enough matches to use SIFT, attempting image registration\n'))
    metric = registration.metric.MattesMutualInformation;
    optimizer = registration.optimizer.RegularStepGradientDescent;
       
    metric.NumberOfSpatialSamples = 500;
    metric.NumberOfHistogramBins = 50;
       
    optimizer.GradientMagnitudeTolerance = 1.00000e-04;
    optimizer.MinimumStepLength = 1.00000e-05;
    optimizer.MaximumStepLength = 1.00000e-01;
    optimizer.MaximumIterations = 100;
    optimizer.RelaxationFactor = 0.5;

    %tform = imregtform(mov_img,ref_img,'translation',optimizer,metric,'PyramidLevels',2);
    tform = affine2d([1 0 0; 0 1 0; 0 0 1]);
end
end