function linClassMdl = linearClassification(data,labels,boxConstraint, labelColumn)
    disp('Linear class');
    linClassMdl = fitcsvm(data,labels(:, labelColumn),'KernelFunction','linear', 'BoxConstraint',boxConstraint);
end
