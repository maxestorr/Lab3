function rbfClassMdl = rbfClassification(data, labels, boxConstraint, labelColumn, sigma)
    rbfClassMdl = fitcsvm(data, labels(:, labelColumn), 'KernelFunction', 'gaussian', ...
        'BoxConstraint', boxConstraint, 'KernelScale', sigma);
end
