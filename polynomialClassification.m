function polynomialClassMdl = polynomialClassification(data,labels,boxConstraint, labelColumn, q)
    %q = polynomial order hyperparameter
    disp('poly');
    polynomialClassMdl = fitcsvm(data, labels(:, labelColumn), 'KernelFunction', 'polynomial', ...
        'BoxConstraint', boxConstraint, 'PolynomialOrder', q);
end
