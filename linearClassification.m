function linClassMdl = linearClassification(data,labels,boxConstraint)
    linClassMdl = fitcsvm,labels, 'KernelFunction','linear', 'BoxConstraint',boxConstraint);
end

