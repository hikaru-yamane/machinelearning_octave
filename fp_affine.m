function Out = fp_affine(In, W, b)
  
  % In��Ҕ�
  X = In;
  
  % ���͂�4�����̂Ƃ���2�����ɕϊ�
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  Out = X*W + b;
  
endfunction