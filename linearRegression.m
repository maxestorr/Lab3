function regMdl = linearRegression(data, targets, boxConstraint, labelColumn, epsilonValue)
disp('Linear reg');
regMdl = fitrsvm(data, targets(:, 1), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue);
end
