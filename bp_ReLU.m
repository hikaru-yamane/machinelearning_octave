function dIn = bp_ReLU(dOut, In, Out)
  % sigmoid�Ƃ̌݊����̂��߁C�֐��̓��͂��]��
  
  Mask = In>0;
  dIn = dOut.*Mask;
  
endfunction