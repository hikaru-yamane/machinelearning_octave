function dIn = bp_ReLU(dOut, In, Out)
  % sigmoid‚Æ‚ÌŒÝŠ·«‚Ì‚½‚ßCŠÖ”‚Ì“ü—Í‚ª—]•ª
  
  Mask = In>0;
  dIn = dOut.*Mask;
  
endfunction