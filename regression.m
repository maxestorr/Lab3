clear all
clc
%% Import Data
[classData, labels, regData, targets] = get_data();
%% Linear Regression test
regMdl = fitrsvm(classData, labels(:,1), 'KernelFunction', 'linear', 'BoxConstraint', 1);
