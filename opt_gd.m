function [Out, V, H] = opt_gd(In, dIn, V, H, iter, alpha, a1, a2)
  
  Out = In - alpha*dIn;
  
endfunction