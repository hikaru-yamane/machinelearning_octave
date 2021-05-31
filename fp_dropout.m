function [Out, Mask] = fp_dropout(In, dropout_ratio, trained_flg)
  
  % ‰Šú‰»
  Mask = 0;
  
  if trained_flg == 0
     Mask = rand(size(In)) > dropout_ratio;
     Out = In.*Mask;
  else
     Out = In*(1 -  dropout_ratio);
  endif
  
endfunction