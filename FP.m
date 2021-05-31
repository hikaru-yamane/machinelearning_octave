classdef FP
  methods(Static)
    
    function Out = affine(In, W, b)
      Out = In*W + b;
    endfunction
    
    function Out = sigmoid(In)
      Out = (1)./(1 + exp(-In)); % 1.0ÇÕÇ»Ç∫ÅH 1./Ç…ÇµÇƒïsà¿íËÇ…Ç»Ç¡ÇΩ
    endfunction
    
    function Out = ReLU(In)
      Out = max(In, 0);
    endfunction
    
    function [Out, J] = sigmoidWithLoss(In, Y, W2, W3, lambda)
      m = size(In, 1);
      
      Out = (1)./(1 + exp(-In));
      J = (-1/m)*sum(sum( Y.*log(Out) + (1 - Y).*log(1 - Out) ));
      % ê≥ë•âª
      J = J + (lambda/(2*m))*sum(sum( W2.^2 )) ...
            + (lambda/(2*m))*sum(sum( W3.^2 ));
    endfunction
    
    function [Out, J] = softmaxWithLoss(In, Y, W2, W3, lambda)
      m = size(In, 1);
      
      Out = exp(In)./sum(exp(In), 2); % â°Ç…ë´Ç∑
      J = (-1/m)*sum(sum( Y.*log(Out) + (1 - Y).*log(1 - Out) ));
      % ê≥ë•âª
      J = J + (lambda/(2*m))*sum(sum( W2.^2 )) ...
            + (lambda/(2*m))*sum(sum( W3.^2 ));
    endfunction
    
    function J = sample1(In)
      J = sum(In);
    endfunction
    
  endmethods
endclassdef
