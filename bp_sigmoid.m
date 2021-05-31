function dIn = bp_sigmoid(dOut, In, Out)
  % ReLUとの互換性のため，関数の入力が余分
  
  dIn = dOut.*( Out.*(1 - Out) );
  
endfunction