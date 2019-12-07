function [accuracyArray] = innerFoldHyperParameterAdjust(test_features, test_labels, modelSelection)
    k_sliceNum = 10;
    [new_features, new_labels] = kFold(k_sliceNum, test_features, test_labels);
    accuracyArray = [];
    for i = 1 : k_sliceNum
        feature_test = new_features(:,:, i);
        label_test = new_labels(:, i);
        clearvars feature_train label_train
     
        %params hold values that are loaded in
        switch(modelSelection)
            case 1 %linear regression
                params = [1,  5, 10, 100, Inf];
            case 2 %linear classification
                params = [1,  5, 10, 100, Inf];
            otherwise
                warning('modelSelection param not properly assigned. Pls check.')
        end

        %k fold is done here with test data from main
        for n = 1 : length(params)
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

            %depending on model selection, paramter is loaded in
            switch(modelSelection)
                case 1
                    linearReg = linearRegression(feature_train, label_train, 1, params(n), 1);
                    predictions = predict(linearReg, feature_test);
                    %L2 error used here, gimme ideas bois
                    accuracyArray = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                case 2
                    linearClass = linearClassification(feature_train, label_train, params(n));
                    predictions = predict(linearClass, feature_test);
                    accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
                case 3
                    polyReg = polynomialRegression(feature_train, label_train, 1, params(n), 2);
                    predictions = predict(polyReg, feature_test);
                    accuracyArray = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                case 4
                    polyClass = polynomialClassification(feature_train, label_train, params(n));
                    predictions = predict(polyClass, feature_test);
                    accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
                case 5
                    rbfReg = rbfRegMdl(feature_train, label_train, 1, params(n), 1, 1);
                    predictions = predict(rbfReg, feature_test);
                    accuracyArray = (1 / 2 * length(label_test)) * sumsqr(predictions - label_test);
                case 6
                    rbfClass = rbfClassMdl(feature_train, label_train, params(n));
                    predictions = predict(rbfClass, feature_test);
                    accuracyArray = [accuracyArray, sum(predictions == label_test) / (size(test_features, 1) / k_sliceNum)];
                otherwise
                    warning('modelSelection param not properly assigned. Pls check.')
            end
        end
    end
end

%This is in case we wanna use gradient descent to automate our parameter
%selection. we don't need to do it but hey.
function [output] = Cost(pred, actual, guess)
    output = ((1/2 *(length(pred)))*sum((pred*guess'-actual).^2));

end

function [output] = dCost(pred, actual, guess)
    output =(1/(length(pred))*(pred'*(pred*(guess)'-actual)));
end