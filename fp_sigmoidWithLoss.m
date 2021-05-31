function [Out, J] = fp_sigmoidWithLoss(In, Y, W2, W3, lambda)
  
  m = size(In, 1);
  
  Out = (1)./(1 + exp(-In));
  J = (-1/m)*sum(sum( Y.*log(Out) + (1 - Y).*log(1 - Out) ));
  
  % ê≥ë•âª
  J = J + (lambda/(2*m))*sum(sum( W2.^2 )) ...
        + (lambda/(2*m))*sum(sum( W3.^2 ));
  
endfunction