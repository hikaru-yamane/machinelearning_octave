function dIn = bp_dropout(dOut, Mask)
  % trained_flg==1�̂Ƃ���bp�Ȃ�
  
  dIn = dOut.*Mask;
  
endfunction