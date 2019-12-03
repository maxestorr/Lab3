function regMdl = linearRegression(data, targets, boxConstraint, epsilonValue)
regMdl = fitrsvm(data, targets(:,1), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
