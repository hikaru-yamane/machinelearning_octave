function checkGradient() % ���z�m�F
  % �Ȃ���dW2�Cdb2�Cdgamma2�Cdbeta2�ŉ�͔����Ɛ��l��������v���Ȃ����Ƃ���D�e�w���Ȃ��Ƃ��P�͔��΂��Ă�̂ɁD�ۂߌ덷�ł��Ȃ�����
  % ���K�����C��(BP)�ɖ�肪����͂��D����dZ2��dgamma2�����v���Ȃ��Ȃ�
  
  fprintf('gradient checking ...\n');
  
  % ������
  X = rand(3, 2); % m>=3�ɂ��Ȃ��Ɛ��K���Ō덷��
##  Y = rand(3, 2);
  Y = [ones(3, 1) zeros(3, 1)]; % softmax�̏���(����ɋC�Â������Ȃ��J����)
  m = size(X, 1);
  W2 = rand(2, 2); % randn���Ɣ����Ƀ[���������Ȃ�m�F���ɂ���
  b2 = zeros(1, 2); % zeros���Ɣ����Ƀ[���������Ȃ�m�F���ɂ���
  W3 = rand(2, 2);
  b3 = zeros(1, 2);
  gamma2 = ones(1, 2);
  beta2 = ones(1, 2)*3; % 2�ȉ��ɂ���ƌ��z��������
  lambda = 0;
  fp_actfunc = @fp_ReLU;
  bp_actfunc = @bp_ReLU;
  fp_lastlayer = @fp_softmaxWithLoss;
  bp_lastlayer = @bp_softmaxWithLoss;
  
  % ��͔���
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
  
  % ���l����
  epsilon = 1e-4; % e�͑g�ݍ��ݕϐ�
  params_numerical = [W2(:); b2(:); gamma2(:); beta2(:); W3(:); b3(:)]; % �u:�v: 2��ڂ�1��ڂ̉�
  dparams_numerical = zeros(size(params_numerical));
  for i = 1:length(params_numerical)
      temp = params_numerical(i);
      % J_plus
      params_numerical(i) = temp + epsilon;
      W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: �u:�v�̋t�̑��� numel: �v�f�̌��̑��a
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
      W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: �u:�v�̋t�̑��� numel: �v�f�̌��̑��a
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
  
  % ��͔����Ɛ��l�����̍�
  disp([dparams_numerical dparams]);
  gap = norm(dparams_numerical - dparams) ...
       /norm(dparams_numerical + dparams);
  fprintf('gap(less than 1e-9): %g\n', gap); % �u%g�v: %f���������̌���\��
  
endfunction