function Out = fp_affine(In, W, b)
  
  % In‚ð‘Ò”ð
  X = In;
  
  % “ü—Í‚ª4ŽŸŒ³‚Ì‚Æ‚«‚Í2ŽŸŒ³‚É•ÏŠ·
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  Out = X*W + b;
  
endfunction