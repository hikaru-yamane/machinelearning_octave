function [Out, V, H] = opt_adam(In, dIn, V, H, iter, alpha, a1, a2)
  
  % èâä˙âª
  epsilon = 1e-8;
  
  V = a1*V + (1-a1)*dIn;
  H = a2*H + (1-a2)*(dIn.^2);
  
  Vhat = V/(1 - a1^iter);
  Hhat = H/(1 - a2^iter);
  
  Out = In - alpha*Vhat./( (Hhat+epsilon).^(0.5) );
  
endfunction