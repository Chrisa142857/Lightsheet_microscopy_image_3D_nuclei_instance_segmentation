function ct = classify_cells_svm(centroids,stable,config)
%--------------------------------------------------------------------------
% Classify cell-types using a trained SVM classifier
%--------------------------------------------------------------------------

% Define outlier classes
classify_by_annotation = false;
outlierTypes = 1:9;
if ~isempty(config.keep_classes)
    outlierTypes = outlierTypes(~ismember(outlierTypes,config.keep_classes));
else
    outlierTypes = [];
end

% Read classifications + patch feautre info
itable = readtable(fullfile(config.output_directory,'classifier',...
    sprintf('%s_patch_info.csv',config.sample_id)));
ftable = readtable(fullfile(config.output_directory,'classifier',...
    sprintf('%s_patch_features.csv',config.sample_id)));
ctable = readtable(fullfile(config.output_directory,'classifier',...
    sprintf('%s_classifications.csv',config.sample_id)));
% Check if merging groups
if ~isempty(config.load_groups)
    fprintf("Merging annotation data by groups \n")
    [~,group_out,group_samples] = get_group_directories(config.sample_id,config.load_groups);

    for i = 1:length(group_out)
        if string(group_samples{i}) == string(config.sample_id)
            continue
        end
        % Read classifications + patch feautre info
        itable = vertcat(itable,readtable(fullfile(group_out{i},'classifier',...
            sprintf('%s_patch_info.csv',group_samples{i}))));
        ftable = vertcat(ftable,readtable(fullfile(group_out{i},'classifier',...
            sprintf('%s_patch_features.csv',group_samples{i}))));
        ctable = vertcat(ctable,readtable(fullfile(group_out{i},'classifier',...
            sprintf('%s_classifications.csv',group_samples{i}))));
    end
end

itable = itable(ctable{:,1} > 0,:);
ftable = ftable(ctable{:,1} > 0,:);
ctable = ctable(ctable{:,1} > 0,:);

% Check table sizes
assert(size(ctable,2) == 1,"Classification table contains more than 1 column")
n_classified = height(ctable);
fprintf("Using %d cells for training SVM model \n",n_classified)
assert(height(ftable) == n_classified, "Number of rows in the feature table do not match")
assert(height(itable) == n_classified, "Number of rows in the info table do not match")

% Combine manual classifications with patch features
ctable.Properties.VariableNames = {'Type'};
if classify_by_annotation
    ftable.Layer = num2str(ftable.Layer);
    ftable.Structure = num2str(ftable.Structure);
else
    % Remove any remaining structure columns
    c1_idx = sprintf("Max_%s",config.markers(config.classify_channels(1)));
    c1_idx = find(ftable.Properties.VariableNames == c1_idx,1);
    ftable = ftable(:,c1_idx:end);
end
trainingData = horzcat(ctable,ftable);

% Train the model
trainedClassifier = trainClassifier(trainingData, outlierTypes);

% Classify centroids in centroid list
ct = zeros(size(centroids,1),1);

% Get layer and structure info
% structures = num2str(bin_annotation_structures(centroids(:,4),'cortex'));
% layers = num2str(bin_annotation_structures(centroids(:,4),'layers'));

% Get patch features for remaining centroids
% stable = get_patch_features(centroids,path_table,config);

% Generate table
feature_table = array2table(stable,'VariableNames',ftable.Properties.VariableNames);

% Run prediction
fprintf("Running prediction... \n")
ct = trainedClassifier.predictFcn(feature_table);

% Remove false positives based on intensity threshold
if ~isempty(config.min_class_thresh)
    low_idx = zeros(size(centroids,1),size(centroids,2)-4);
    for i = 5:size(centroids,2)
        thresh = prctile(centroids(:,i),config.min_class_thresh*100);
        low_idx(:,i-4) = centroids(:,i)<thresh;
    end
    low_idx = all(low_idx,2);
    ct(low_idx) = 1;
end

end


function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData, outlierTypes)

warning('off','stats:cvpartition:KFoldMissingGrp')

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = inputTable.Properties.VariableNames(2:end);
predictors = inputTable(:, predictorNames);
response = inputTable.Type;
isCategoricalPredictor = logical(varfun(@ischar,inputTable,'output','uniform'));
classnames = unique(inputTable.Type);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
%%%%%%%%%%%%%%%%%%%%
template = templateSVM(...
    'KernelFunction', 'linear', ... 
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsall', ...
    'ClassNames', classnames);
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%
% lambda = 1.4821e-08;
% layerSizes = [2350 1360 2390];
% classificationSVM = fitcnet(...
%   predictors, response, ...
%   'Standardize', false, ...
%   'LayerSizes',layerSizes, ...
%   'Activations','tanh', ...
%   'Lambda',lambda);
%%%%%%%%%%%%%%%%%%%%

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

% Compute validation predictions
[validationPredictions, score] = kfoldPredict(partitionedModel);

% Display validation accuracy
%validationPredictions(ismember(validationPredictions,outlierTypes)) = min(outlierTypes);
%response(ismember(response,outlierTypes)) = min(outlierTypes);
save('/cajal/ACMUSERS/ziquanw/Lightsheet/coloc_numorph/numorph_src/coloc_annotation_p4.mat', 'validationPredictions')
fprintf("Correctly classified %.2f%% of cells after cross validation \n",...
    sum(validationPredictions==response)*100/length(response))

for r = unique(response)'
    idx = (response == r);
    fprintf("Correctly classified %.2f%% of class-%02d cells \n",...
        sum(validationPredictions(idx)==response(idx))*100/length(response(idx)), r)
end


%responses = unique(response)';
%for i = 1:length(responses)
%fprintf("Classifying at rate of %.2f%% of true predictions for class %d \n",...
%    sum(validationPredictions==responses(i))*100/sum(response == i),responses(i))
%end

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

end



