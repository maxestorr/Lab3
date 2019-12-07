function [bestParam] = innerFoldHyperParameterAdjust(test_features, test_labels, modelSelection)
    k_sliceNum = 10;
    [new_features, new_labels] = kFold(k_sliceNum, test_features, test_labels);
    avgAcc = [];
    accuracyArray = [];
    
    % k-fold
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
    end
    
    %depending on model selection, paramter is loaded in
    switch(modelSelection)
        case 1
            for constraint = 10:100:10
                for epsilon = 1:10:1
                    linearReg = linearRegression(feature_train, label_train, epsilon, constraint, 1);
                    predictions = predict(linearReg, feature_test);
                    %L2 error used here, gimme ideas bois
                    accuracyArray = [accuracyArray, (1 / 2 * length(label_test)) * sumsqr(predictions - label_test)];
                end
            end
        case 2
            for boxConstraint = 10:100:10                
                linearClass = linearClassification(feature_train, label_train, boxConstraint);
                predictions = predict(linearClass, feature_test);
                accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
            end
        case 3
            for constraint = 10:100:10
                for epsilon = 1:10:1
                    for polyOrder = 1:5:1
                        polyReg = polynomialRegression(feature_train, label_train, constraint, epsilon, 1, polyOrder);
                        predictions = predict(polyReg, feature_test);
                        accuracyArray = [accuracyArray, (1 / 2 * length(label_test)) * sumsqr(predictions - label_test)];
                    end
                end
            end
        case 4
            for constraint = 10:100:10
                for polyOrder = 1:5:1
                    polyClass = polynomialClassification(feature_train, label_train, constraint, polyOrder, 1);
                    predictions = predict(polyClass, feature_test);
                    accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
                end
            end
        case 5
            for constraint = 10:100:10
                for epsilon = 1:10:1
                    for sigma = 1:10:1
                        rbfReg = rbfRegMdl(feature_train, label_train, constraint, epsilon, 1, sigma);
                        predictions = predict(rbfReg, feature_test);
                        accuracyArray = [accuracyArray, (1 / 2 * length(label_test)) * sumsqr(predictions - label_test)];
                    end
                end
            end
        case 6
            for constraint = 10:100:10
                for sigma = 1:10:1
                    rbfClass = rbfClassMdl(feature_train, label_train, constraint, sigma, 1);
                    predictions = predict(rbfClass, feature_test);
                    accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
                end
            end
        otherwise
            warning('modelSelection param not properly assigned. Pls check.')
    end
avgAcc = [avgAcc, mean(accuracyArray)];
accuracyArray = [];
best_index = find(avgAcc == max(avgAcc(:)));
bestParam = params(best_index(1));
end


%This is in case we wanna use gradient descent to automate our parameter
%selection. we don't need to do it but hey.
function [output] = Cost(pred, actual, guess)
    output = ((1/2 *(length(pred)))*sum((pred*guess'-actual).^2));

end

function [output] = dCost(pred, actual, guess)
    output =(1/(length(pred))*(pred'*(pred*(guess)'-actual)));
end