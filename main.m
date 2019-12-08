clear all
clc
modelSelection = 6;
labelColumn = 1;

[classData, labels, regData, targets] = getData();
disp("Data loaded");

regData = regData(1:1000, :);

if mod(modelSelection, 2) == 1
    targets = targets(1:1000, :);
else 
    targets = labels(1:1000, :);
end

k_sliceNum = 10;
[new_features, new_labels] = kFold(k_sliceNum, regData, targets);

accuracyArray = zeros(k_sliceNum, 1);
fscoreArray = zeros(k_sliceNum, 1);
total_fMeasureScore = 0;

for i = 1 : k_sliceNum 
    feature_test = new_features(:,:, i);
    label_test = new_labels(:, i);
    clearvars feature_train label_train
    for j = 1 : k_sliceNum
        if i ~= j
            if exist('feature_train','var') == 1
                feature_train = cat(1, feature_train, new_features(:,:, j));
            else
                feature_train = new_features(:,:, j);
            end
            if exist('label_train','var') == 1
                label_train = cat(1, label_train, new_labels(:, j));
            else
                label_train = new_labels(:, j);
            end
        end
    end
    switch(modelSelection)
        case 1           
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn);    
            boxConstraint = bestParam(1,1);
            epsilon = bestParam(1,2);
            model = linearRegression(feature_train, label_train, boxConstraint, labelColumn, epsilon);
            predictions = predict(model, feature_test);
            
            difference = predictions - label_test;
            squaredError = difference .^ 2;
            meanSquaredError = sum(squaredError(:)) / numel(label_test);
            rmsError = sqrt(meanSquaredError);

            accuracyArray(i) = rmsError;
        case 2
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn);     
            boxConstraint = bestParam(1,1);
            model = linearClassification(feature_train, label_train, boxConstraint, labelColumn);
            predictions = predict(model, feature_test);
            accuracyArray(i) = sum(predictions == label_test) / (size(targets, 1) / k_sliceNum);
        
        case 3
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn);
            boxConstraint = bestParam(1,1);
            epsilon = bestParam(1,2);
            polyOrder = bestParam(1,3);           
            model = polynomialRegression(feature_train, label_train, boxConstraint, epsilon, labelColumn, polyOrder);
            predictions = predict(model, feature_test);
            difference = predictions - label_test;
            squaredError = difference .^ 2;
            meanSquaredError = sum(squaredError(:)) / numel(label_test);
            rmsError = sqrt(meanSquaredError);
            accuracyArray(i) = rmsError;
        case 4
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn);   
            boxConstraint = bestParam(1,1);
            polyOrder = bestParam(1,2);           
            model = polynomialClassification(feature_train, label_train, boxConstraint, labelColumn, polyOrder);
            predictions = predict(model, feature_test);
            accuracy = sum(predictions == label_test) / (size(targets, 1) / k_sliceNum);
        
        case 5
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn); 
            boxConstraint = bestParam(1,1);
            epsilon = bestParam(1,2);
            sigma = bestParam(1,3);               
            model = rbfRegression(feature_train, label_train, boxConstraint, epsilon, labelColumn, sigma);
            predictions = predict(model, feature_test);
            difference = predictions - label_test;
            squaredError = difference .^ 2;
            meanSquaredError = sum(squaredError(:)) / numel(label_test);
            rmsError = sqrt(meanSquaredError);
            accuracyArray(i) = rmsError;
        case 6
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, modelSelection, labelColumn); 
            boxConstraint = bestParam(1,1);
            sigma = bestParam(1,2);                      
            model = rbfClassification(feature_train, label_train, boxConstraint, labelColumn, sigma);
            predictions = predict(model, feature_test);
            accuracyArray(i) = sum(predictions == label_test) / (size(targets, 1) / k_sliceNum);
    end
end

disp(bestParam)
disp(mean(accuracyArray))
disp(length(model.SupportVectors));
disp(length(model.SupportVectors) / length(model.BoxConstraints));