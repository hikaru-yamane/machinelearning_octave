function acc = predict(X, y, fp_actfunc, ...
                       W2, b2, W3, b3, W4, b4, ...
                       gamma2, beta2, gamma3, beta3, ...
                       pad, stride, ph2, pw2, stride2, ...
                       f1_2, f6_2, f1_3, f6_3, ...
                       dropout_ratio, ...
                       trained_flg, leaned_flg);
  
  if leaned_flg == 1
     % âÊëúï`âÊ(2éüå≥)
     imagesc(X(:,:,:,1)); colorbar;
  endif
  
  % êÑò_
  A1 = X;
  C2 = fp_convolution(A1, W2, b2, pad, stride);
  N2 = fp_normalization(C2, gamma2, beta2, f1_2, f6_2, leaned_flg);
  A2 = fp_actfunc(N2);
  P2 = fp_pooling(A2, ph2, pw2, pad, stride2);
  Z3 = fp_affine(P2, W3, b3);
  N3 = fp_normalization(Z3, gamma3, beta3, f1_3, f6_3, leaned_flg);
  A3 = fp_actfunc(N3);
  D3 = fp_dropout(A3, dropout_ratio, trained_flg);
  Z4 = fp_affine(D3, W4, b4);
  [maxVal, maxInd] = max(Z4, [], 2); % â°Ç…ç≈ëÂíl
  acc = mean( maxInd==y )*100;
  
  if leaned_flg == 1
     fprintf('predicted number: %d\n', maxInd(1));
  endif
  
endfunction