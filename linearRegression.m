function regMdl = linearRegression(data, targets, epsilonValue)
regMdl = fitrsvm(data, targets(:,1), 'KernelFunction', 'linear', 'BoxConstraint', 1, ...
'Epsilon', epsilonValue);
end