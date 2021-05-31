function dIn = bp_softmaxWithLoss(dJ, Out, Y)
  
  m = size(Out, 1);
  
  dIn = (1/m)*(Out - Y);
  
endfunction