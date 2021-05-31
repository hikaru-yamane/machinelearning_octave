function dIn = bp_ReLU(dOut, In, Out)
  % sigmoidとの互換性のため，関数の入力が余分
  
  Mask = In>0;
  dIn = dOut.*Mask;
  
endfunction