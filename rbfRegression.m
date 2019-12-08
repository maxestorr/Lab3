function rbfRegMdl = rbfRegression(data, targets, boxConstraint, labelColumn, epsilonValue, sigma)
rbfRegMdl = fitrsvm(data, targets, 'KernelFunction', 'gaussian', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue, 'KernelScale', sigma);
end
