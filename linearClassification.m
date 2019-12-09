function linClassMdl = linearClassification(data,labels,boxConstraint)
    linClassMdl = fitcsvm(data,labels,'KernelFunction','linear',...
        'BoxConstraint',boxConstraint);
end

