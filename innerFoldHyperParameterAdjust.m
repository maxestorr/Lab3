function [bestParam] = innerFoldHyperParameterAdjust(test_features, test_labels, modelSelection)
    k_sliceNum = 10;
    [new_features, new_labels] = kFold(k_sliceNum, test_features, test_labels);
    paramEval = []
    
    % k-fold (NEEDS REIMPLEMENTING SINCE I FUCKED IT)
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
            for constraint = 10:10:100
                for epsilon = 1:1:10
                    linearReg = linearRegression(feature_train, label_train, epsilon, constraint, 1);
                    predictions = predict(linearReg, feature_test);
                    %L2 error used here, gimme ideas bois
                    acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                    paramEval = [paramEval; constraint, epsilon, acc];
                end
            end
        case 2
            for constraint = 10:10:100                
                linearClass = linearClassification(feature_train, label_train, constraint);
                predictions = predict(linearClass, feature_test);
                acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                paramEval = [paramEval; constraint, acc];
            end
        case 3
            for constraint = 10:10:100
                for epsilon = 1:1:10
                    for polyOrder = 1:1:5
                        polyReg = polynomialRegression(feature_train, label_train, constraint, epsilon, 1, polyOrder);
                        predictions = predict(polyReg, feature_test);
                        acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                        paramEval = [paramEval; constraint, epsilon, polyOrder, acc];
                    end
                end
            end
        case 4
            for constraint = 10:10:100
                for polyOrder = 1:1:5
                    polyClass = polynomialClassification(feature_train, label_train, constraint, polyOrder, 1);
                    predictions = predict(polyClass, feature_test);
                    acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                    paramEval = [paramEval; constraint, polyOrder, acc];
                end
            end
        case 5
            for constraint = 10:10:100
                for epsilon = 1:1:10
                    for sigma = 1:1:10
                        rbfReg = rbfRegMdl(feature_train, label_train, constraint, epsilon, 1, sigma);
                        predictions = predict(rbfReg, feature_test);
                        acc = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                        paramEval = [paramEval; constraint, epsilon, sigma, acc];
                    end
                end
            end
        case 6
            for constraint = 10:10:100
                for sigma = 1:1:10
                    rbfClass = rbfClassMdl(feature_train, label_train, constraint, sigma, 1);
                    predictions = predict(rbfClass, feature_test);
                    acc = sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum);
                    paramEval = [paramEval; constraint, sigma, acc];
                end
            end
        otherwise
            warning('modelSelection param not properly assigned. Pls check.')
    end

% Find row of parameters for best accuracy
[bestRows, ~] = find(paramEval(:, end) == max(paramEval(:, end)));
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