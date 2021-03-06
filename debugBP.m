function debugBP(In)
  
  % 初期化
  X = In; % 入力は変更できないため待避
  Y = [ones(4,1) zeros(4,3)];
  W2 = 0;
  W3 = 0;
  lambda = 0;
  dJ = 0;
  
  % 解析微分
##  Out = fp_sigmoid(X);
  [Out, f1, f3, f6, f7, f8, f9, f11] = fp_normalization(X, 1, 0, 0, 0, 0);
  J = sum(sum(Out)); % 仮の損失関数
  dOut = ones(size(Out));
##  dIn = bp_sigmoid(dOut, X, Out);
  [dIn, dgamma, dbeta] = bp_normalization(dOut, f3, f6, f7, f8, f9, f11);
  
  % 数値微分
  epsilon = 1e-4;
  dIn_num = zeros(size(X));
  for i = 1:size(X, 1)
  for j = 1:size(X, 2)
    temp = X(i, j);
    X(i, j) = temp + epsilon;
##    Out = fp_sigmoid(X);
    [Out, f1, f3, f6, f7, f8, f9, f11] = fp_normalization(X, 1, 0, 0, 0, 0);
    J_plus = sum(sum(Out));
    X(i, j) = temp - epsilon;
##    Out = fp_sigmoid(X);
    [Out, f1, f3, f6, f7, f8, f9, f11] = fp_normalization(X, 1, 0, 0, 0, 0);
    J_minus = sum(sum(Out));
    dIn_num(i, j) = (J_plus - J_minus)/(2*epsilon);
    X(i, j) = temp;
  endfor
  endfor
  
  disp([dIn_num - dIn](:));
  
endfunction
