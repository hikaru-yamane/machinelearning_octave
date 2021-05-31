function [dIn, dW, db] = bp_affine(dOut, In, W, lambda)
  
  % In‚ğ‘Ò”ğ
  X = In;
  
  m = size(dOut, 1);
  
  % “ü—Í‚ª4ŸŒ³‚Ì‚Æ‚«‚Í2ŸŒ³‚É•ÏŠ·
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  dIn = dOut*W';
  dW = X'*dOut;
  db = sum(dOut, 1); % c‚É‘«‚·
  
  % ³‘¥‰»
  dW = dW + (lambda/m)*W;
  
  % dIn‚ğ4ŸŒ³‚É•ÏŠ·
  if ndims(In) == 4
     dIn = reshape(dIn', size(In));
  endif
  
endfunction