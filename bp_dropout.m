function dIn = bp_dropout(dOut, Mask)
  % trained_flg==1のときはbpなし
  
  dIn = dOut.*Mask;
  
endfunction