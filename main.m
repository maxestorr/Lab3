clear all
clc
modelSelection = 1;

[classData, labels, regData, targets] = getData();
disp("Data loaded");

regData = regData(1:1000, :);
targets = targets(1:1000, :);
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
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, 1);           
            linearReg = linearRegression(feature_train, label_train, 1, bestParam, 1);
            predictions = predict(linearReg, feature_test);
            accuracy = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
        case 2
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, 2);           
            linearClass = linearClassification(feature_train, label_train, bestParam );
            predictions = predict(linearClass, feature_test);
            accuracy = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
        case 3
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, 3);           
            polyReg = polynomialRegression(feature_train, label_train, 1, bestParam, 1);
            predictions = predict(polyReg, feature_test);
            accuracy = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
        case 4
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, 2);           
            rbfReg = rbfRegression(feature_train, label_train, bestParam );
            predictions = predict(rbfReg, feature_test);
            accuracy = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
        case 5
            bestParam = innerFoldHyperParameterAdjust(feature_train, label_train, 2);           
            rbfClass = rbfClassMdl(feature_train, label_train, bestParam );
            predictions = predict(rbfClass, feature_test);
            accuracy = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
    end
end