clear all
clc

[classData, labels, regData, targets] = getData();
disp("Data loaded");

features = regData(1:1000, :);
labels = targets(1:1000, :);

k_sliceNum = 10;
[new_features, new_labels] = kFold(k_sliceNum, features, labels);

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

    linearReg = linearRegression(regData, targets, 0.1);
    
    %predictions = testdt(tree, feature_test);
    %[fMeasureScore, precisionValue, recallValue] = fMeasure(label_test, predictions);
    
    %total_fMeasureScore = total_fMeasureScore + fMeasureScore;
    %fscoreArray(i) = fMeasureScore;
    %accuracyArray(i) = sum(predictions == label_test) / (size(features, 1) / k_sliceNum);
    

end