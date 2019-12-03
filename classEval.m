function accuracy = classEval[classMdl,testData,testLabels]
  fit = predict(classMdl,testData));
  accuracy = 100*(1-sum((fit - testLabels).^2)/(length(testLabels));
end
