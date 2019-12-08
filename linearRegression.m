function regMdl = linearRegression(data, targets, boxConstraint, labelColumn, epsilonValue)
regMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
