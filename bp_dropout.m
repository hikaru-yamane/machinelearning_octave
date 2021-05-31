function dIn = bp_dropout(dOut, Mask)
  % trained_flg==1‚Ì‚Æ‚«‚Íbp‚È‚µ
  
  dIn = dOut.*Mask;
  
endfunction