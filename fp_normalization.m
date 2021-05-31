function [Out, f1, f3, f6, f7, f8, f9, f11] = fp_normalization(In, gamma, beta, f1L, f6L, leaned_flg)
  % �ǐ����ቺ���邩��܂Ƃ߂Ȃ�
  % lean 0:���_ 1:�w�K
  
  % �Ҕ�
  X = In;
  
  % ���͂�4�����̂Ƃ���2�����ɕϊ�
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  % ������
  [m, n] = size(X);
  epsilon = 1e-8; % �I�[�o�[�t���[�΍�10e-7
  
  f0 = X; % qitta�̐}���Q�l�ɂ��Ă邩��Y�������[������
  f1 = sum(f0, 1)/m; % �c�ɑ���
  if leaned_flg == 1
     f1 = f1L;
  endif
  f2 = ones(m, n).*f1;
  f3 = f0 - f2;
  f4 = f3.^2;
  f5 = sum(f4, 1)/m;
  f6 = (f5 + epsilon).^(0.5);
  if leaned_flg == 1
     f6 = f6L;
  endif
  f7 = (1)./f6;
  f8 = ones(m, n).*f7;
  f9 = f3.*f8;
  f10 = gamma;
  f11 = ones(m, n).*f10;
  f12 = f11.*f9;
  f13 = beta;
  f14 = ones(m, n).*f13;
  Out = f12 + f14;
  
  % 4�����ɕϊ�
  if ndims(In) == 4
     Out = reshape(Out', size(In));
  endif
  
endfunction