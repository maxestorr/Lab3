clear all
clc

[classData, labels, regData, targets] = getData();
disp("Data loaded");
%%
[newData,newLabels] = kFold(10,classData,labels);
newData = newData(1:100,:,:);
newLabels = newLabels(1:100,:,:);
classAcc = zeros(10,20);
supportVectorAbs = zeros(10,20);
supportVectorPer = zeros(10,20);
for i = 1:10
    for j = 1:20
        testData = newData(:,:,i);
        testLabels = newLabels(:,i);
        trainData = newData;
        trainData(:,:,i) = [];
        trainData = permute(trainData,[1 3 2]);
        trainData = reshape(trainData,900,98);
        trainLabels = newLabels;
        trainLabels(:,i) = [];
        trainLabels = trainLabels(:);
        model = linearClassification(trainData,trainLabels,10^((0.5*(j-15))));
        classAcc(i,j) = classEval(model,testData,testLabels);
        supportVectorAbs(i,j) = length(model.SupportVectors(:,1))
        supportVectorPer(i,j) = supportVectorAbs(i,j)/900
        i,j
    end
end
%%
[newReg, newTargets] = kFold(10,regData,targets);
regErr = zeros(10,10,10);
newReg = newReg(1:100,:,:);
newTargets = newTargets(1:100,:,:);
supportVectorAbs = zeros(10,10,10);
supportVectorPer = zeros(10,10,10);
for i = 1:10
    i
    for j = 1:10
        j
        for k = 1:10
            testReg = newReg(:,:,i);
            testTargets = newTargets(:,i);
            trainReg = newReg;
            trainReg(:,:,i) = [];
            trainReg = permute(trainReg,[1 3 2]);
            trainReg = reshape(trainReg,900,98);
            trainTargets = newTargets;
            trainTargets(:,i) = [];
            trainTargets = trainTargets(:);
            trainReg = permute(trainReg,[1 3 2]);
            trainReg = reshape(trainReg,900,98);
            model = linearRegression(trainReg,trainTargets,10^(j-7),10^(k-7));
            regErr(i,j,k) = regEval(model,testReg,testTargets);
            supportVectorAbs(i,j) = length(model.SupportVectors(:,1))
            supportVectorPer(i,j) = supportVectorAbs(i,j)/900
            k
        end
    end
end
