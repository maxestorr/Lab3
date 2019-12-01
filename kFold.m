function [newFeatures, newLabels] = kFold(k, features, labels)
    dataLength = length(labels);
    A = 1:dataLength;
    A = A(randperm(length(A)));
        
    remainderValue = mod(dataLength, k);
    A = A(1: dataLength-remainderValue);
    
    indicesArray = reshape(A, [], k);
    
    newFeatures = zeros(length(indicesArray),size(features,2), k);
    newLabels = zeros(length(indicesArray), k);

    for i = 1:k
        for j = 1:length(indicesArray)
            for k = 1:size(features,2)
                newFeatures(j,k,i) = features(indicesArray(j, i), k);
            end
            newLabels(j,i) = labels(indicesArray(j,i));
        end
    end
end
