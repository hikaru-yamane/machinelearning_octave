function dIn = bp_sigmoid(dOut, In, Out)
  % ReLU‚Æ‚ÌŒİŠ·«‚Ì‚½‚ßCŠÖ”‚Ì“ü—Í‚ª—]•ª
  
  dIn = dOut.*( Out.*(1 - Out) );
  
endfunction