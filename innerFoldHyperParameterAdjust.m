function [bestParams] = innerFoldHyperParameterAdjust(test_features, test_labels, modelSelection, labelColumn)
    k_sliceNum = 10;
    [new_features, new_labels] = kFold(k_sliceNum, test_features, test_labels);
    paramEval = [];
    
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
            for constraint = 0.1:0.1:1
                for epsilon = 0.1:0.1:1
                    linearReg = linearRegression(feature_train, label_train, epsilon, labelColumn, constraint);
                    predictions = predict(linearReg, feature_test);
                    %L2 error used here for cost
                    acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                    paramEval = [paramEval; constraint, epsilon, acc];
                end
            end
        case 2
            for constraint = 0.1:0.1:1               
                linearClass = linearClassification(feature_train, label_train, constraint, labelColumn);
                predictions = predict(linearClass, feature_test);
                acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                paramEval = [paramEval; constraint, acc];
            end
        case 3
            for constraint = 0.1:0.1:1
                for epsilon = 0.1:0.1:1
                    for polyOrder = 1:1:5
                        polyReg = polynomialRegression(feature_train, label_train, constraint, labelColumn, epsilon, polyOrder);
                        predictions = predict(polyReg, feature_test);
                        acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                        paramEval = [paramEval; constraint, epsilon, polyOrder, acc];
                    end
                end
            end
            
        case 4
            for constraint = 0.1:0.1:1
                for polyOrder = 1:1:5
                    polyClass = polynomialClassification(feature_train, label_train, constraint, labelColumn, polyOrder);
                    predictions = predict(polyClass, feature_test);
                    acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                    paramEval = [paramEval; constraint, polyOrder, acc];
                end
            end
        case 5
            for constraint = 0.1:0.1:1
                for epsilon = 0.1:0.1:1
                    for sigma = 150:10:250
                        rbfReg = rbfRegression(feature_train, label_train, constraint, labelColumn, epsilon, sigma);
                        predictions = predict(rbfReg, feature_test);
                        acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                        paramEval = [paramEval; constraint, epsilon, sigma, acc];
                    end
                end
            end
        case 6
            for constraint = 0.1:0.1:1
                for sigma = 150:10:250
                    rbfClass = rbfClassification(feature_train, label_train, constraint, labelColumn, sigma);
                    predictions = predict(rbfClass, feature_test);
                    acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                    paramEval = [paramEval; constraint, sigma, acc];
                end
            end
        otherwise
            warning('modelSelection param not properly assigned. Pls check.')
    end

% Find row of parameters for best accuracy
[bestRows, ~] = find(paramEval(:, end) == min(paramEval(:, end)));
bestParams = paramEval(bestRows(1), 1:end-1);  % Exclude accuracy value

end


%This is in case we wanna use gradient descent to automate our parameter
%selection. we don't need to do it but hey.
function [output] = Cost(pred, actual, guess)
    output = ((1/2 *(length(pred)))*sum((pred*guess'-actual).^2));

end

function [output] = dCost(pred, actual, guess)
    output =(1/(length(pred))*(pred'*(pred*(guess)'-actual)));
end
