function dIn = bp_sigmoidWithLoss(dJ, Out, Y)
  
  m = size(Out, 1);
  
  dIn = (1/m)*(Out - Y);
  
endfunction