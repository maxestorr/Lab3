clear all
clc
%% Import Data
[classData, labels, regData, targets] = getData();
%% Linear Classification test
classMdl = fitcsvm(classData((1:74610),:),labels((1:74610),1), 'KernelFunction','linear', 'BoxConstraint',1);
fit = predict(classMdl,classData((74611:82906),:));
accuracy = 100*(1-sum((fit - labels((74611:82906),1)).^2)/(8295));