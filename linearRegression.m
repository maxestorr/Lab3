function regMdl = linearRegression(data, targets, boxConstraint, epsilonValue, labelColumn)
regMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
