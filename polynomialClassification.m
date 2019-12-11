function polynomialClassMdl = polynomialClassification(data,labels,boxConstraint, labelColumn, q)
    %q = polynomial order hyperparameter
    disp('poly');
    polynomialClassMdl = fitcsvm(data, labels(:, 1), 'KernelFunction', 'polynomial', ...
        'BoxConstraint', boxConstraint, 'PolynomialOrder', q);
end
