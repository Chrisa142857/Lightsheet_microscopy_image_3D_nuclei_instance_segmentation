function ct = classify_cells_threshold(centroids, config)
%--------------------------------------------------------------------------
% Classify cell-types using intensity thresholding
%--------------------------------------------------------------------------
remove_l1 = 'false';
remove_background = 'false';

% Get parameters
markers = config.markers;
n_markers = length(markers);

annotations = centroids.annotations;
intensities = centroids.intensities;
centroids = centroids.coordinates;

% Calculate class combinations from the number of markers
if isequal(config.contains_nuclear_channel,"true")
    ref_n = 2;
    n = n_markers-1;
else
    ref_n = 1;
    n = n_markers;
end
classes = dec2bin(2^n-1:-1:0)-'0';

% Check threshold lengths
if ~isempty(config.intensity_thresholds)
    n_thresh = length(ref_n:n_markers);
    if length(config.intensity_thresholds) ~= n_thresh
        error("Expected %d threshold values but received %d",...
            n_thresh, length(config.intensity_thresholds))
    end
end
    
% Create empty vector for storing cell-types
ct = ones(size(centroids,1),1);

% Remove any centroids without annotations
rm_idx = annotations == 0;
fprintf('%s\t Removed %d nuclei with no annotation \n',datetime('now'),sum(rm_idx))

% Remove cells below minimum nuclei intensity threshold
if isequal(config.contains_nuclear_channel,"true")
    low_idx = intensities(:,1) < config.min_intensity;
    rm_idx = rm_idx | low_idx;
    fprintf('%s\t Removed %d low intensity nuclei \n',datetime('now'),sum(low_idx))
end

% Ask if to remove layer 1 prior to analysis
if isequal(remove_l1,"true")
   l1_table = readtable(fullfile(config.home_path,'annotations','cortical_layers',...
       'layer1.csv'));
   l1_idx = ismember(centroids(:,4),l1_table.index);
  
   s1 = centroids(:,6) > prctile(centroids(l1_idx,6),75);
   s2 = centroids(:,7) > prctile(centroids(l1_idx,7),75);

   l1_idx = s1 & s2 & l1_idx;
   rm_idx = rm_idx | l1_idx;
   fprintf('%s\t Removed %d cells \n',datetime('now'),sum(l1_idx))
end

% Remove selected indexes
centroids = centroids(~rm_idx,:);
intensities1 = intensities(~rm_idx,:);

% Intitialize matrices and models
count = zeros(size(centroids,1),n_markers);
a = ref_n-1;
for i = ref_n:n_markers
    % Clustering using thresholds
    intensities = intensities1(:,i);

    % Load background intensities from background images
    if isequal(remove_background,'true')
       fprintf('%s\t Subtracting background from background image \n',datetime('now'))
        % If given background images, get background intensities from these
        %[back_thresh, ~] = get_background_at_centroids(centroids,...
        %    config,B{i},S{i});
        back_vals = readmatrix(fullfile(config.output_directory,'background','back_values.csv'));
        back_vals = back_vals(~rm_idx,:);
        back_thresh = back_vals(:,i-1);
        
    else
        % Otherwise use constant background
        back_thresh = 0;
    end
    
    % Subtract background
    values = intensities - back_thresh;
    values(values<0) = 0;
    
    % Calculate threshold
    if ~isempty(config.intensity_thresholds)
        % From manually specified value
        if config.intensity_thresholds(i-a) > 0 && config.intensity_thresholds(i-a) < 1
            thresh = config.intensity_thresholds(i-a)*65535;
        elseif config.intensity_thresholds(i-a) > 1 
            thresh = config.intensity_thresholds(i-a);
        else
            thresh = 0;
        end
        fprintf('%s\t Using manual threshold %d on marker %s \n',...
            datetime('now'),thresh, markers(i))

    else
        % From expression
        if iscell(config.intensity_expression)
            expression = config.intensity_expression{i-1};
        else
            expression = config.intensity_expression;
        end
        thresh = calculate_threshold_expression(values, expression);
        fprintf('%s\t Using expression threshold %d on marker %s \n',...
            datetime('now'),thresh, markers(i))
    end
    
    % Apply z normalization
    if isequal(config.z_normalization,"true")
        fprintf('%s\t Applying z normalization \n',datetime('now'))
        thresh = (thresh - mean(values))/std(values);
        values = (values - mean(values))/std(values);
    end
    
    % Count cells above threshold
    count(:,i) = values>thresh;
end

if ref_n == 2
    count = count(:,2:end);
end
[~,idx] = sort(sum(classes,2));
classes = classes(idx,:);

% Calculate classes
[~,c] = ismember(count,classes,'rows');
ct(~rm_idx) = c;
ct(rm_idx) = 0;

end


function threshold = calculate_threshold_expression(values, expression)
%--------------------------------------------------------------------------
% Calculate threshold from defined value or regular expression
%--------------------------------------------------------------------------

if isempty(expression)
    return
end

% Deconstruct regular expression
splitstring = regexp(expression,'\s','split');

threshold = 0;
for i = 1:length(splitstring)
    s1 = regexp(splitstring(i),'*','split');
    
    % Constant to multiply by
    if length(s1) < 2
        c = 1;
    else
        c = str2double(s1(1));
        s1(1) = [];
    end
    
    % Some stat from data
    s2 = regexp(s1,'(\w+)','match');
    if isempty(s2)
        continue
    end
    switch s2(1)
        case 'mean'
            stat = mean(values);
        case 'median'
            stat = median(values);
        case 'mode'
            stat = mode(values);
        case 'std'
            stat = std(values);
        case 'mad'
            stat = mad(values,1);
        case 'prctile'
            stat = prctile(values,str2double(s2(2)));
        case '+'
            continue
        otherwise
            warning("Could not recognize regular expression term %d\n",i)
            continue
    end
    % Add to threshold value
    threshold = threshold + stat*c;
end

end