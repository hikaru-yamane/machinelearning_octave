function dIn = bp_sigmoid(dOut, In, Out)
  % ReLU�Ƃ̌݊����̂��߁C�֐��̓��͂��]��
  
  dIn = dOut.*( Out.*(1 - Out) );
  
endfunction