function [Out, V, H] = opt_adagrad(In, dIn, V, H, iter, alpha, a1, a2)
  
  % ‰Šú‰»
  epsilon = 1e-8;
  
  H = H + dIn.^2;
  Out = In - alpha*dIn./( (H+epsilon).^(0.5) );
  
endfunction