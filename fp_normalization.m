function [Out, f1, f3, f6, f7, f8, f9, f11] = fp_normalization(In, gamma, beta, f1L, f6L, leaned_flg)
  % 可読性が低下するからまとめない
  % lean 0:推論 1:学習
  
  % 待避
  X = In;
  
  % 入力が4次元のときは2次元に変換
  if ndims(X) == 4
     [h, w, c, m] = size(X);
     X = reshape(X, [c*w*h, m])';
  endif
  
  % 初期化
  [m, n] = size(X);
  epsilon = 1e-8; % オーバーフロー対策10e-7
  
  f0 = X; % qittaの図を参考にしてるから添え字がゼロから
  f1 = sum(f0, 1)/m; % 縦に足す
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
  
  % 4次元に変換
  if ndims(In) == 4
     Out = reshape(Out', size(In));
  endif
  
endfunction