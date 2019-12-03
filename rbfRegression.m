function rbfRegMdl = rbfRegression(data, targets, boxConstraint, epsilonValue, labelColumn, sigma)
rbfRegMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue, 'KernelScale', sigma);
end
