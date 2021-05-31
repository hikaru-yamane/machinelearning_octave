function debugBP_cnn(In) % 4 4 1 2
  
  % ‰Šú‰»
  X = In; % “ü—Í‚Í•ÏX‚Å‚«‚È‚¢‚½‚ß‘Ò”ğ
##  W = rand(2,2,2,4);
##  b = rand(1,4);
##  ph = 2;
##  pw = 2;
##  stride = 2;
  gamma = 1;
  beta = 0;
  dropout_ratio = 0.3;
  trained_flg = 0;
  
  % ‰ğÍ”÷•ª
  [Out, Mask] = fp_dropout(X, dropout_ratio, trained_flg);
##  [Out, f11, f9, f8, f7, f6, f3] = fp_normalization(X, gamma, beta);
##  [Out, Mask] = fp_pooling(X, ph, pw, pad=0, stride);
##  [Out, In_two, W_two] = fp_convolution(X, W, b, pad=0, stride=1);
##  Out = fp_ReLU(X);
  J = sum(sum(sum(sum(Out)))); % ‰¼‚Ì‘¹¸ŠÖ”
  dOut = ones(size(Out));
  dIn = bp_dropout(dOut, Mask);
##  [dIn, dgamma, dbeta] = bp_normalization(dOut, f11, f9, f8, f7, f6, f3);
##  dIn = bp_pooling(dOut, X, Mask, ph, pw, pad=0, stride);
##  [dIn, dW, db] = bp_convolution(dOut, X, In_two, W, W_two, pad=0, stride=1);
##  dIn = bp_ReLU(dOut, X);
  
  % ”’l”÷•ª
  epsilon = 1e-4;
  dIn_num = zeros(size(X));
  for i = 1:size(X, 1)
  for j = 1:size(X, 2)
  for k = 1:size(X, 3)
  for l = 1:size(X, 4)
    temp = X(i, j, k, l);
    X(i, j, k, l) = temp + epsilon;
    [Out, Mask] = fp_dropout(X, dropout_ratio, trained_flg);
##    [Out, f11, f9, f8, f7, f6, f3] = fp_normalization(X, gamma, beta);
##    [Out, Mask] = fp_pooling(X, ph, pw, pad=0, stride);
##    [Out, In_two, W_two] = fp_convolution(X, W, b, pad=0, stride=1);
##    Out = fp_ReLU(X);
    J_plus = sum(sum(sum(sum(Out))));
    X(i, j, k, l) = temp - epsilon;
    [Out, Mask] = fp_dropout(X, dropout_ratio, trained_flg);
##    [Out, f11, f9, f8, f7, f6, f3] = fp_normalization(X, gamma, beta);
##    [Out, Mask] = fp_pooling(X, ph, pw, pad=0, stride);
##    [Out, In_two, W_two] = fp_convolution(X, W, b, pad=0, stride=1);
##    Out = fp_ReLU(X);
    J_minus = sum(sum(sum(sum(Out))));
    dIn_num(i, j, k, l) = (J_plus - J_minus)/(2*epsilon);
    X(i, j, k, l) = temp;
  endfor
  endfor
  endfor
  endfor
  
  disp([dIn_num - dIn](:));
##  disp([dIn_num - dIn]);
  
endfunction
