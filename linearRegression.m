function regMdl = linearRegression(data, targets, boxConstraint, epsilonValue)
regMdl = fitrsvm(data, targets, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
