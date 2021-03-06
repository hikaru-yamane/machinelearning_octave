function [dIn, dW, db] = bp_affine(dOut, In, W, lambda)
  
  % Inを待避
  X = In;
  
  m = size(dOut, 1);
  
  % 入力が4次元のときは2次元に変換
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  dIn = dOut*W';
  dW = X'*dOut;
  db = sum(dOut, 1); % 縦に足す
  
  % 正則化
  dW = dW + (lambda/m)*W;
  
  % dInを4次元に変換
  if ndims(In) == 4
     dIn = reshape(dIn', size(In));
  endif
  
endfunction