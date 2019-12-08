function [fMeasureScore, precisionValue, recallValue] = fMeasure(actual, predicted)
    %Implementing F1 where β = 1
    
    precisionValue = precision(actual, predicted);
    recallValue = recall(actual, predicted);
    
    fMeasureScore = 2 * ((precisionValue * recallValue) / (precisionValue + recallValue));
end

function output = precision(actual, predicted)
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i = 1:length(actual)
        if actual(i) == 1
            if predicted(i) == 1
                TP = TP + 1;
            else
                FN = FN + 1;
            end
        else
            if predicted(i) == 1
                FP = FP +1;
            else
                TN = TN + 1;
            end
        end 
    end
    precision = TP/(TP + FP);
    if(isnan(precision))
        output = 0;
    else
        output = precision;
    end
end

function output = recall(actual, predicted)
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i = 1:length(actual)
        if actual(i) == 1
            if predicted(i) == 1
                TP = TP + 1;
            else
                FP = FP + 1;
            end
        else
            if predicted(i) == 1
                FN = FN +1;
            else
                TN = TN + 1;
            end
        end
                
            
    end
    recall = TP/(TP + FN);
    if(isnan(recall))
        output = 0;
    else
        output = recall;
    end
end
