clear all
clc
%% Import Data
[classData, labels, regData, targets] = getData();
%% Linear Classification test
classMdl = fitcsvm(classData, labels(:,1), 'KernelFunction', 'linear', 'BoxConstraint', 1);
