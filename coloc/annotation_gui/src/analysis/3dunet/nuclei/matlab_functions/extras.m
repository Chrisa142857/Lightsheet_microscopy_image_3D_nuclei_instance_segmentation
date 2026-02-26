%% Calculate Splits/Mergers by Sorting with Nearest Centroid
n_true = length(cen_truth);
n_predict = length(cen_predict);

cen_count_true = zeros(n_true,3);
cen_count_true(:,1) = 1:n_true;
cen_count_true(:,3) = NaN;
cen_count_predict = zeros(n_predict,3);
cen_count_predict(:,1) = 1:n_predict;

for i = 1:n_predict
    % Position of predicted centroid
    pos = cen_predict(i,:);    
    % Measure distance from every true centroid
    D = sqrt(sum((pos - cen_truth).^2, 2));
    % Find nearest distance to true centroid
    [min_d,idx] = min(D);
    % Set the index for this as being truly selected
    cen_count_true(idx,2) = cen_count_true(idx,2) + 1;
    % Set nearest distance
    cen_count_true(idx,3) = min(min_d,cen_count_true(idx,3));
    % Set index matched true centroid 
    cen_count_predict(i,2) = cen_count_true(idx,1);
    % Save distance to centroid
    cen_count_predict(i,3) = min_d;
end

%% Trim nuclei around edges (After Assigning)
padding = [3, 3, 2];
img_end = img_size - padding;

idx_x = cen_truth(:,1)>padding(1) & cen_truth(:,1) < img_end(1);
idx_y = cen_truth(:,2)>padding(2) & cen_truth(:,2) < img_end(2);
idx_z = cen_truth(:,3)>padding(3) & cen_truth(:,3) < img_end(3);

idx_final = find(idx_x & idx_y & idx_z);
cen_count_true_final = cen_count_true(idx_final,:);
cen_count_predict_final = cen_count_predict(ismember(cen_count_predict(:,2), idx_final),:);

n_true = size(cen_count_true_final,1);
n_predict = size(cen_count_predict_final,1);
nan_rows = isnan(cen_count_true_final(:,3));

fprintf('\nAverage Nearest Distance: %d\n',mean(cen_count_true_final(~nan_rows,3)))
fprintf('Merger Rate: %d\n',sum(cen_count_true_final(:,2) == 0)*100/n_true)
fprintf('Split Rate: %d\n',sum(cen_count_true_final(:,2)>1)*100/n_true)