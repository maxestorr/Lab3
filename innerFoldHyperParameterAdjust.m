function [Alpha] = innerFoldHyperParameterAdjust(predictions, actual, learning_rate)
    iter = 1;
    Maxiter = 10;
    Theta = 3;
    Tol = 10;
    Alpha = learning_rate

Bestfunc = zeros(Maxiter, 1);
Err = zeros(Maxiter,1);
FunErr = zeros(Maxiter,1);
DummyVar = Theta;
DummyVarf = Cost(predictions, actual,Theta);

for iter = 1:Maxiter
      Theta = Theta - Alpha*(Der_Cost(predictions, actual, Theta))';
      Bestfunc(iter)= Cost(predictions, actual, Theta);
      Err(iter) = norm(Theta-DummyVar);
      FunErr(iter) = abs((Bestfunc(iter)-DummyVarf)/Bestfunc(iter));
      FunTol = 3;
             
    if (Err(iter)< Tol)
        if (FunErr(iter)< FunTol)
        fprintf('\n -----CHANGE IN THETA & COST FUNCTION LESS THAN SPECIFIED TOLERANCE-----\n');
        r = (1-(-1)).*rand(1,1) +(-1);
        Alpha = abs(Alpha + r);
        break;
        end
    end
    
    if (iter == Maxiter)
        fprintf('\n -----MAXIMUM NUMBER OF ITERATIONS REACHED-----\n');
        break;
    end
end
end

function [output] = Cost(pred, actual, guess)
    output = ((1/(2*20))*sum((pred*guess'-actual).^2));

end

function [output] = Der_Cost(pred, actual, guess)
    output =(1/(2 * length(pred))*(pred'*(pred*(guess)'-actual)));

end