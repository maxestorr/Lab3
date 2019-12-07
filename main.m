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

epsilonValue = 0.1;

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
            linearReg = linearRegression(feature_train, label_train, 1, epsilonValue, 1);
            predictions = predict(linearReg, feature_test);
            alpha = innerFoldHyperParameterAdjust(feature_test, label_test, 1);
        case 2
            linearReg = linearClassification(feature_train, label_train, epsilonValue );
            predictions = predict(linearReg, feature_test);
            alpha = innerFoldHyperParameterAdjust(feature_test, label_test, 2);
        case 3
            linearReg = linearRegression(feature_train, label_train, 1, epsilonValue, 1);
            predictions = predict(linearReg, feature_test);
            alpha = innerFoldHyperParameterAdjust(feature_test, label_test, 1);
    end

    %total_fMeasureScore = total_fMeasureScore + fMeasureScore;
    %fscoreArray(i) = fMeasureScore;
    %accuracyArray(i) = sum(predictions == label_test) / (size(features, 1) / k_sliceNum);
    

end