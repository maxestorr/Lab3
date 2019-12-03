function accuracy = classEval(classMdl,testData,testLabels)
  fit = predict(classMdl,testData);
  accuracy = 100 * (1 - abs(sum(fit - testLabels) / length(testLabels)));
end
