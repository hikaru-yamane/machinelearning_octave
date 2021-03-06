function Out = fp_affine(In, W, b)
  
  % Inを待避
  X = In;
  
  % 入力が4次元のときは2次元に変換
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  Out = X*W + b;
  
endfunction