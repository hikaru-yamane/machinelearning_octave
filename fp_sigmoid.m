function Out = fp_sigmoid(In)
  
  % 1./にして不安定になった
  Out = (1)./(1 + exp(-In)); % 1.0はなぜ？
  
endfunction