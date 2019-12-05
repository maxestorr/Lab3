function [Alpha] = innerFoldHyperParameterAdjust(predictions, actual, learning_rate)
    iter = 1;
    maxIterations = 10;
    Theta = 3;
    Alpha = learning_rate;

bestParam = zeros(maxIterations, 1);

for iter = 1:maxIterations
      Theta = Theta - Alpha*(Der_Cost(predictions, actual, Theta))';
      bestParam(iter)= Cost(predictions, actual, Theta);
      %get err
    
    if (iter == maxIterations)
        break;
    end
end
end

function [output] = Cost(pred, actual, guess)
    output = ((1/2 *(length(pred)))*sum((pred*guess'-actual).^2));

end

function [output] = Der_Cost(pred, actual, guess)
    output =(1/(length(pred))*(pred'*(pred*(guess)'-actual)));

end