clear all
clc

[classData, labels, regData, targets] = getData();
disp("Data loaded");

regData = regData(1:1000, :);
targets = targets(1:1000, :);

linearReg = linearRegression(regData, targets, 0.1);