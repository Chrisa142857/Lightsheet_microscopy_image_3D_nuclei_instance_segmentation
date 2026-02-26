%% Calculate Splits/Mergers By Counting Centroids In Cell Volumes
function [mult, miss, merge, empty, correct, total_error, mult_lbls, miss_lbls, merge_lbls] = evaluate_by_fill(cen_predict,label,prediction)

% Get unqiue labels
uniq_lbls = unique(label);
uniq_lbls = uniq_lbls(uniq_lbls > 0);
n_label = length(uniq_lbls);

% Get indexes for centroids
cen_predict_idx = zeros(1,size(cen_predict,1));
cen_predict = round(cen_predict);
for i = 1:length(cen_predict_idx)
    cen_predict_idx(i) = sub2ind(size(label),cen_predict(i,1),cen_predict(i,2),...
        cen_predict(i,3));
end

% Generate label matrix for storing counts
lbl_count_true = zeros(n_label,3);
lbl_count_true(:,1) = 1:n_label;

% Check each label for number of predicted centroids
for i = 1:n_label
    % Pixels for label
    idx = find(label == uniq_lbls(i));
    % Count predicted centroids overlapping pixel indexes
    cen_count = length(intersect(idx,cen_predict_idx));
    % Check if totally missing
    total_pixels = sum(prediction(idx));
    if total_pixels == 0 
        cen_count = -1;
    end
    % Save
    lbl_count_true(i,2) = cen_count;
    % Save label
    lbl_count_true(i,3) = uniq_lbls(i);
end

% Check for centroids that don't ovrelap any label
empty_idx = 0;
for i = 1:length(cen_predict_idx)
    idx = cen_predict_idx(i);
    if label(idx) == 0
        empty_idx = empty_idx + 1;
    end
end

% Calculate how many labels are missing centroids or have multiple
mult = sum(lbl_count_true(:,2) > 1);
merge = sum(lbl_count_true(:,2) == 0);
miss = sum(lbl_count_true(:,2) < 0);
empty = empty_idx;
correct = sum(lbl_count_true(:,2) == 1);
total_error = 1 - correct/(correct + mult+merge+miss);

mult_lbls = lbl_count_true(lbl_count_true(:,2) > 1,:);
merge_lbls = lbl_count_true(lbl_count_true(:,2) == 0,:);
miss_lbls = lbl_count_true(lbl_count_true(:,2)< 0,:);

fprintf('Multiple Centroids: %f\n',mult/n_label)
fprintf('Missing Centroids: %f\n',miss/n_label)
fprintf('Merged Centroids: %f\n',merge/n_label)
fprintf('Empty Centroids: %f\n',empty/length(cen_predict_idx))
fprintf('Total Error: %f\n',total_error)

end
