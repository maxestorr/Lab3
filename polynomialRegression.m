function polynomialRegMdl = polynomialRegression(data, targets, boxConstraint, labelColumn, epsilonValue, q)
    %q = polynomial order hyperparameter
    polynomialRegMdl = fitrsvm(data, targets(:, labelColumn), 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint, ...
                 'Epsilon', epsilonValue, 'PolynomialOrder', q);
end
