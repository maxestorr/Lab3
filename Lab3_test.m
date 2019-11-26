clear all
clc
%% Import Data
[classData, labels, regData, targets] = get_data();
%% Linear Classification test
classMdl = fitcsvm(classData,labels(:,1), 'KernelFunction','linear', 'BoxConstraint',1);