function polynomialClassMdl = polynomialClassification(data,labels,boxConstraint, labelColumn, q)
    %q = polynomial order hyperparameter
    polynomialClassMdl = fitcsvm(data, labels(:, labelColumn), 'KernelFunction', 'gaussian', ...
        'BoxConstraint', boxConstraint, 'PolynomialOrder', q);
end
