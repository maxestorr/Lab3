fit = predict(classMdl,classData((74611:82906),:));
accuracy = 100*(1-sum((fit - labels((74611:82906),1)).^2)/(8295));
