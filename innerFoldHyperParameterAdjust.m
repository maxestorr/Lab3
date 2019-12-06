function [Alpha] = innerFoldHyperParameterAdjust(test_features, test_labels, learning_rate)
    k_sliceNum = 10;
    [new_features, new_labels] = kFold(k_sliceNum, test_features, test_labels);
    
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
        
    linearReg = linearRegression(regData, targets, alpha);
    predictions = predict(linearReg, feature_test);
    alpha = innerFoldHyperParameterAdjust(feature_test, label_test, alpha);
    
    accuracyArray(i) = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);

    %total_fMeasureScore = total_fMeasureScore + fMeasureScore;
    %fscoreArray(i) = fMeasureScore;
    %accuracyArray(i) = sum(predictions == label_test) / (size(features, 1) / k_sliceNum);
    

    end
end

function [output] = Cost(pred, actual, guess)
    output = ((1/2 *(length(pred)))*sum((pred*guess'-actual).^2));

end

function [output] = Der_Cost(pred, actual, guess)
    output =(1/(length(pred))*(pred'*(pred*(guess)'-actual)));

end