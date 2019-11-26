function [classFeatures, labels, regFeatures, targets] = getData()
    classX = table2array(readtable('predx_for_classification.csv'));
    classY = table2array(readtable('predy_for_classification.csv'));
    labels = table2array(readtable('label.csv'));
    classFeatures = [classX,classY];
    regX = table2array(readtable('predx_for_regression.csv'));
    regY = table2array(readtable('predy_for_regression.csv'));
    targets = table2array(readtable('angle.csv'));
    regFeatures = [regX,regY];
end
