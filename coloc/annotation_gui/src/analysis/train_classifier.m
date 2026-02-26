function classif = train_classifier(classif, use_classes, use_features)
%--------------------------------------------------------------------------
% Classify cell-types using a trained SVM classifier within app
%--------------------------------------------------------------------------

if nargin<3 
    use_features = classif.featureNames;
end

% Subset only patches that were annotated
rm_idx = classif.classes == 0;
trainingData = horzcat(classif.classes(~rm_idx)',classif.features(~rm_idx,:));

% Get outlier types
u_classes = unique(trainingData(:,1));
outlierTypes = u_classes(~ismember(u_classes, use_classes));

% Get features
idx = cellfun(@(s) startsWith(s, strcat(use_features,"_")), classif.featureNames);
trainingData = trainingData(:,[true, idx]);
featureNames = classif.featureNames(idx);

% Train the model
[trainedClassifier, validationPredictions] = trainSVMClassifier(trainingData(:,1),...
    trainingData(:,2:end), featureNames);

[validationAccuracy, confusionmatrix] = get_accuracy(trainingData(:,1), validationPredictions, outlierTypes);

% Add to classifier structure
classif.method = "svm";
classif.classifier = trainedClassifier;
classif.validationAccuracy = validationAccuracy;
classif.confusionmatrix = confusionmatrix;

end


function [trainedClassifier, validationPredictions] = trainSVMClassifier(response, predictors, predictorNames)

warning('off','stats:cvpartition:KFoldMissingGrp')

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
classnames = unique(string(response));

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
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

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;

% Perform cross-validation
kfolds = min(100, length(response));
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', kfolds);

% Compute validation predictions
validationPredictions = cellfun(@(s) str2double(s), kfoldPredict(partitionedModel));

end


function [validationAccuracy, C] = get_accuracy(response, validationPredictions, outlierTypes)

% Calculate confusion matrix
C = confusionmat(response, validationPredictions);

% Merge outlier types
if ~isempty(outlierTypes)
    response(ismember(response, outlierTypes)) = outlierTypes(1);
    validationPredictions(ismember(validationPredictions, outlierTypes)) = outlierTypes(1);
end
validationAccuracy = sum(validationPredictions == response)/length(response);

end
