function [Out, V, H] = opt_momentum(In, dIn, V, H, iter, alpha, a1, a2)
  
  V = a1*V - alpha*dIn;
  Out = In + V;
  
endfunction