function [dIn, dgamma, dbeta] = bp_normalization(dOut, f3, f6, f7, f8, f9, f11)
  % dIn�͂܂Ƃ߂Ȃ��D�ŏI�I�ɃL���C�ɂȂ�Ȃ����C�ǐ����ቺ���邩��
  
  % �Ҕ�
  X = dOut;
  
  % ���͂�4�����̂Ƃ���2�����ɕϊ�
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  % ������
  [m, n] = size(X);
  
  d15 = X;
  d14 = sum(d15, 1); % �c�ɑ���
  d12a = d15.*f11;
  d12b = d15.*f9;
  d11 = sum(d12b, 1);
  d9a = d12a.*f8;
  d9b = d12a.*f3;
  d8 = sum(d9b, 1);
  d7 = -d8.*(f7.^2); % �O�̂���(-f7.^2)�͔����Ƃ�
  d6 = d7.*((1)./(2*f6));
  d5 = ones(m, n).*d6./m;
  d4 = d5.*(2*f3);
  d3a = d9a + d4;
  d3b = -d3a;
  d2 = sum(d3b, 1);
  d1 = ones(m, n).*d2./m;
  dIn = d3a + d1;
  
  dgamma = d11;
  dbeta = d14;
  
  % 4�����ɕϊ�
  if ndims(dOut) == 4
     dIn = reshape(dIn', size(dOut));
  endif
  
endfunction