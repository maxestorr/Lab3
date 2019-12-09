function polynomialRegMdl = polynomialRegression(data, targets, boxConstraint, labelColumn, epsilonValue, q)
    %q = polynomial order hyperparameter
    disp('poly');
    polynomialRegMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'polynomial', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue, 'PolynomialOrder', q);
end
