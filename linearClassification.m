function linClassMdl = linearClassification(data,labels,boxConstraint, labelColumn)
    linClassMdl = fitcsvm(data,labels(:, labelColumn,'KernelFunction','linear',...
        'BoxConstraint',boxConstraint));
end
