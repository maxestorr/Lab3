function regMdl = linearRegression(data, targets, boxConstraint, labelColumn, epsilonValue)
disp('Linear reg');
regMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
