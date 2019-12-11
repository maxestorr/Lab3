function rbfClassMdl = rbfClassification(data, labels, boxConstraint, labelColumn, sigma)
    disp('rbf');
    rbfClassMdl = fitcsvm(data, labels(:, 1), 'KernelFunction', 'gaussian', ...
        'BoxConstraint', boxConstraint, 'KernelScale', sigma);
end
