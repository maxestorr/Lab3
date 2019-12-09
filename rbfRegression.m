function rbfRegMdl = rbfRegression(data, targets, boxConstraint, labelColumn, epsilonValue, sigma)
disp('rbf');
rbfRegMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'gaussian', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue, 'KernelScale', sigma);
end
