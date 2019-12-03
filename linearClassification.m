function linClassMdl = linearClassification(data,labels,boxConstraint)
    linClassMdl = fitcsvm(data((1:74610),:),labels((1:74610),1),...
        'KernelFunction','linear', 'BoxConstraint',boxConstraint);
end

