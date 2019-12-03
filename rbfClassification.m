function rbfClassMdl = rbfClassification(data,labels,boxConstraint, sigma, labelColumn)
    rbfClassMdl = fitcsvm(data, labels(:, labelColumn), 'KernelFunction', 'gaussian', ...
        'BoxConstraint', boxConstraint, 'KernelScale', sigma);
end