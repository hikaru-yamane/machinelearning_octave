function checkGradient() % 勾配確認
  % なぜかdW2，db2，dgamma2，dbeta2で解析微分と数値微分が一致しないことあり．各層少なくとも１つは発火してるのに．丸め誤差でもなさそう
  % 正規化レイヤ(BP)に問題があるはず．いつもdZ2やdgamma2から一致しなくなる
  
  fprintf('gradient checking ...\n');
  
  % 初期化
  X = rand(3, 2); % m>=3にしないと正規化で誤差大
##  Y = rand(3, 2);
  Y = [ones(3, 1) zeros(3, 1)]; % softmaxの条件(これに気づかずかなり苦労した)
  m = size(X, 1);
  W2 = rand(2, 2); % randnだと微分にゼロが多くなり確認しにくい
  b2 = zeros(1, 2); % zerosだと微分にゼロが多くなり確認しにくい
  W3 = rand(2, 2);
  b3 = zeros(1, 2);
  gamma2 = ones(1, 2);
  beta2 = ones(1, 2)*3; % 2以下にすると勾配消失発生
  lambda = 0;
  fp_actfunc = @fp_ReLU;
  bp_actfunc = @bp_ReLU;
  fp_lastlayer = @fp_softmaxWithLoss;
  bp_lastlayer = @bp_softmaxWithLoss;
  
  % 解析微分
  A1      = X;
  Z2      = fp_affine(A1, W2, b2);
  [N2, f11, f9, f8, f7, f6, f3] ...
          = fp_normalization(Z2, gamma2, beta2);
  A2      = fp_actfunc(N2);
  Z3      = fp_affine(A2, W3, b3);
  [A3, J] = fp_lastlayer(Z3, Y, W2, W3, lambda);
  dZ3             = bp_lastlayer(dJ=1, A3, Y);
  [dA2, dW3, db3] = bp_affine(dZ3, A2, W3, lambda);
  dN2             = bp_actfunc(dA2, Z2); % ReLU:(dOut, In), sigmoid:(dOut, Out)
  [dZ2, dgamma2, dbeta2] ...
                  = bp_normalization(dN2, f11, f9, f8, f7, f6, f3);
  [dA1, dW2, db2] = bp_affine(dZ2, A1, W2, lambda);
  dparams = [dW2(:); db2(:); dgamma2(:); dbeta2(:); dW3(:); db3(:)];
  
  % 数値微分
  epsilon = 1e-4; % eは組み込み変数
  params_numerical = [W2(:); b2(:); gamma2(:); beta2(:); W3(:); b3(:)]; % 「:」: 2列目を1列目の下
  dparams_numerical = zeros(size(params_numerical));
  for i = 1:length(params_numerical)
      temp = params_numerical(i);
      % J_plus
      params_numerical(i) = temp + epsilon;
      W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: 「:」の逆の操作 numel: 要素の個数の総和
      index_temp = numel(W2);
      b2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(b2)-1), size(b2));
      index_temp = index_temp + numel(b2);
      gamma2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(gamma2)-1), size(gamma2));
      index_temp = index_temp + numel(gamma2);
      beta2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(beta2)-1), size(beta2));
      index_temp = index_temp + numel(beta2);
      W3 = reshape(params_numerical(index_temp+1:index_temp+1+numel(W3)-1), size(W3));
      index_temp = index_temp + numel(W3);
      b3 = reshape(params_numerical(index_temp+1:end), size(b3));
      A1           = X;
      Z2           = fp_affine(A1, W2, b2);
      [N2, f11, f9, f8, f7, f6, f3] = fp_normalization(Z2, gamma2, beta2);
      A2           = fp_actfunc(N2);
      Z3           = fp_affine(A2, W3, b3);
      [A3, J_plus] = fp_lastlayer(Z3, Y, W2, W3, lambda);
      % J_minus
      params_numerical(i) = temp - epsilon;
      W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: 「:」の逆の操作 numel: 要素の個数の総和
      index_temp = numel(W2);
      b2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(b2)-1), size(b2));
      index_temp = index_temp + numel(b2);
      gamma2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(gamma2)-1), size(gamma2));
      index_temp = index_temp + numel(gamma2);
      beta2 = reshape(params_numerical(index_temp+1:index_temp+1+numel(beta2)-1), size(beta2));
      index_temp = index_temp + numel(beta2);
      W3 = reshape(params_numerical(index_temp+1:index_temp+1+numel(W3)-1), size(W3));
      index_temp = index_temp + numel(W3);
      b3 = reshape(params_numerical(index_temp+1:end), size(b3));
      A1            = X;
      Z2            = fp_affine(A1, W2, b2);
      [N2, f11, f9, f8, f7, f6, f3] = fp_normalization(Z2, gamma2, beta2);
      A2            = fp_actfunc(N2);
      Z3            = fp_affine(A2, W3, b3);
      [A3, J_minus] = fp_lastlayer(Z3, Y, W2, W3, lambda);
      % dparams_numerical
      dparams_numerical(i) = (J_plus - J_minus)/(2*epsilon);
      params_numerical(i) = temp;
  endfor
  
  % 解析微分と数値微分の差
  disp([dparams_numerical dparams]);
  gap = norm(dparams_numerical - dparams) ...
       /norm(dparams_numerical + dparams);
  fprintf('gap(less than 1e-9): %g\n', gap); % 「%g」: %fよりも多くの桁を表示
  
endfunction