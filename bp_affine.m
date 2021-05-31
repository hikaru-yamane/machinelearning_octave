function [dIn, dW, db] = bp_affine(dOut, In, W, lambda)
  
  % In��Ҕ�
  X = In;
  
  m = size(dOut, 1);
  
  % ���͂�4�����̂Ƃ���2�����ɕϊ�
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  dIn = dOut*W';
  dW = X'*dOut;
  db = sum(dOut, 1); % �c�ɑ���
  
  % ������
  dW = dW + (lambda/m)*W;
  
  % dIn��4�����ɕϊ�
  if ndims(In) == 4
     dIn = reshape(dIn', size(In));
  endif
  
endfunction