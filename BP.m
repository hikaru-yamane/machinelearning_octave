classdef BP
  methods(Static)
    
    function [dIn, dW, db] = affine(dOut, In, W, lambda)
      m = size(dOut, 1);
      
      dIn = dOut*W';
      dW = In'*dOut;
      db = sum(dOut, 1); % ècÇ…ë´Ç∑
      % ê≥ë•âª
      dW = dW + (lambda/m)*W;
    endfunction
    
    function dIn = sigmoid(dOut, Out)
      dIn = dOut.*( Out.*(1 - Out) );
    endfunction
    
    function dIn = ReLU(dOut, Out)
      Mask = Out~=0; % sigmoidÇ∆çáÇÌÇπÇÈÇΩÇﬂInÇ≈ÇÕÇ»Ç≠Out
      dIn = dOut.*Mask;
    endfunction
    
    function dIn = sigmoidWithLoss(dJ, Out, Y)
      m = size(Out, 1);
      
      dIn = (1/m)*(Out - Y);
    endfunction
    
    function dIn = softmaxWithLoss(dJ, Out, Y)
      m = size(Out, 1);
      
      dIn = (1/m)*(Out - Y);
    endfunction
    
    function dIn = sample1(dJ, In)
       dIn = ones(size(In));
    endfunction
    
  endmethods
endclassdef