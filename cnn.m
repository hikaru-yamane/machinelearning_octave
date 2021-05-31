% �ϐ���������������; ���ׂẴt�B�M���A����; �R�}���h�E�B���h�E�̃N���A
clear; close all; clc;


% nn�\��
% (20,20,1,m)��c(16,16,30,m)��p(8,8,30,m)
cn2 = 30*16*16;
pn2 = 30*8*8;
n3 = 50;
n4 = 10; % 0��10�ƃ��x��


% �f�[�^�Ǎ� X:5000*400(-0.2-1.2�ɐ��K��), y:5000*1(�x�N�g���C0��10�ƃ��x��)
% .txt���ƓǍ��x������
fprintf('data loading ...\n');
load ex4data1.mat;
##% �P���x�ɕϊ� ���r���Ŕ{���x�ɖ߂��Ă��܂�
##X = single(X);
% �f�[�^����(6:2:2) ������test�g��Ȃ�����8:2
m = size(X, 1);
randInd = randperm(m); % 1:m�������_���ɒu������
Xcv = X(randInd(1:m*2/10), :);
ycv = y(randInd(1:m*2/10), :);
temp = m*2/10;
##Xtest = X(randInd(temp+1:(temp+1+m*2/10)-1), :);
##ytest = y(randInd(temp+1:(temp+1+m*2/10)-1), :);
##temp = temp + m*2/10;
X = X(randInd(temp+1:end), :);
y = y(randInd(temp+1:end), :);
% one-hot�\���ɕϊ�
##Y = zeros(m, output_layer_size);
##for i = 1:output_layer_size
##    Y(:, i) = Y(:, i) + y==i;
##endfor
Y   = y==[1:n4];  % m*10(one-shot�\���C�s��)
Ycv = ycv==[1:n4];
##Ytest = ytest==[1:output_layer_size];
% ���͂�4�����ɕϊ�
m = size(X, 1);
X = reshape(X', [20,20,1,m]);
mcv = size(Xcv, 1);
Xcv = reshape(Xcv', [20,20,1,mcv]);


% �p�����[�^(�d�݂ƃo�C�A�X)�̃����_��������
##% Xavier�̏����l(sigmoid, tanh)
##init_epsilon2 = sqrt(1/input_layer_size);
##init_epsilon3 = sqrt(1/hidden_layer_size);
% He�̏����l(ReLU)
init_epsilon = 0.01;
init_epsilon3 = sqrt(2/pn2);
init_epsilon4 = sqrt(2/n3);
% �K�E�X���z�ŏ�����
W2 = randn(5, 5, 1, 30)*init_epsilon;
b2 = zeros(1, 30);
W3 = randn(pn2, n3)*init_epsilon3;
b3 = zeros(1, n3);
W4 = randn(n3, n4)*init_epsilon4;
b4 = zeros(1, n4);
% �p�����[�^(���K�����C��)�̏�����
gamma2 = ones(1, cn2);
beta2 = zeros(1, cn2);
gamma3 = ones(1, n3);
beta3 = zeros(1, n3);


% �n�C�p�[�p�����[�^
alpha = 0.009;     % �w�K��learning rate 0.01 AdaGrad
lambda = 0.0;    % �������̏d��regularization weight
% conv + pooling
pad = 0;
stride = 1;
ph2 = 2;
pw2 = ph2;
stride2 = 2; % pooling�w��ph��stride����v������(��{)
% optimizer
a1 = 0.9;
a2 = 0.999;
% dropout
dropout_ratio = 0.5; % 0-1
% �J��Ԃ��� 400
minibatch_size = 100; % ��
iter_per_epoch = m/minibatch_size; % 1�G�|�b�N������̉�
epochs_num = 10;
iters_num = epochs_num*iter_per_epoch;


% �w�K
fprintf('training ...\n');
for iter = 1:iters_num
    
    % ������
    if iter == 1
       % �֐�
       fp_actfunc = @fp_ReLU;
       bp_actfunc = @bp_ReLU;
       fp_lastlayer = @fp_softmaxWithLoss;
       bp_lastlayer = @bp_softmaxWithLoss;
       optimizer = @opt_adam;
       % �X�V
       V_W2 = 0;
       V_b2 = 0;
       V_W3 = 0;
       V_b3 = 0;
       V_W4 = 0;
       V_b4 = 0;
       V_gamma2 = 0;
       V_beta2 = 0;
       V_gamma3 = 0;
       V_beta3 = 0;
       H_W2 = 0;
       H_b2 = 0;
       H_W3 = 0;
       H_b3 = 0;
       H_W4 = 0;
       H_b4 = 0;
       H_gamma2 = 0;
       H_beta2 = 0;
       H_gamma3 = 0;
       H_beta3 = 0;
       % �ϐ�
       epoch = 1;
       leaned_flg = 0;
       % ���O
       iter_log = zeros(iters_num, 1);
       J_log = zeros(iters_num, 1);
       epoch_log = zeros(epochs_num, 1);
       acc_log = zeros(epochs_num, 1);
       acccv_log = zeros(epochs_num, 1);
    endif
    
    % �t���O
    trained_flg = 0;
    
    % �~�j�o�b�`����
    randInd = randperm(m);
    Xbatch = X(:, :, :, randInd(1:minibatch_size));
    Ybatch = Y(randInd(1:minibatch_size), :);
    
    % FP
    A1           = Xbatch;
    [C2, A1_two, W2_two] ...
                 = fp_convolution(A1, W2, b2, pad, stride);
    [N2, f1_2, f3_2, f6_2, f7_2, f8_2, f9_2, f11_2] ...
                 = fp_normalization(C2, gamma2, beta2, f1L=0, f6L=0, leaned_flg);
    A2           = fp_actfunc(N2);
    [P2, MaskP2] = fp_pooling(A2, ph2, pw2, pad, stride2);
    Z3           = fp_affine(P2, W3, b3);
    [N3, f1_3, f3_3, f6_3, f7_3, f8_3, f9_3, f11_3] ...
                 = fp_normalization(Z3, gamma3, beta3, f1L=0, f6L=0, leaned_flg);
    A3           = fp_actfunc(N3);
    [D3, MaskD3] = fp_dropout(A3, dropout_ratio, trained_flg);
    Z4           = fp_affine(D3, W4, b4);
    [A4, J]      = fp_lastlayer(Z4, Ybatch, W2, W3, W4, lambda);

    % BP
    dZ4             = bp_lastlayer(dJ=1, A4, Ybatch);
    [dD3, dW4, db4] = bp_affine(dZ4, D3, W4, lambda);
    dA3             = bp_dropout(dD3, MaskD3);
    dN3             = bp_actfunc(dA3, N3, A3);
    [dZ3, dgamma3, dbeta3] = bp_normalization(dN3, f3_3, f6_3, f7_3, f8_3, f9_3, f11_3);
    [dP2, dW3, db3] = bp_affine(dZ3, P2, W3, lambda);
    dA2             = bp_pooling(dP2, A2, MaskP2, ph2, pw2, pad=0, stride2);
    dN2             = bp_actfunc(dA2, N2, A2);
    [dC2, dgamma2, dbeta2] = bp_normalization(dN2, f3_2, f6_2, f7_2, f8_2, f9_2, f11_2);
    [dA1, dW2, db2] = bp_convolution(dC2, A1, A1_two, W2, W2_two, pad=0, stride=1, lambda);
    
    fprintf('Iter: %d, Cost: %f\n', iter, J);
    trained_flg = 1;
    
    % �F�����x
    if rem(iter, iter_per_epoch) == 0
       acc = predict(X, y, fp_actfunc, ...
                     W2, b2, W3, b3, W4, b4, ...
                     gamma2, beta2, gamma3, beta3, ...
                     pad, stride, ph2, pw2, stride2, ...
                     f1_2, f6_2, f1_3, f6_3, ...
                     dropout_ratio, ...
                     trained_flg, leaned_flg);
       acccv = predict(Xcv, ycv, fp_actfunc, ...
                     W2, b2, W3, b3, W4, b4, ...
                     gamma2, beta2, gamma3, beta3, ...
                     pad, stride, ph2, pw2, stride2, ...
                     f1_2, f6_2, f1_3, f6_3, ...
                     dropout_ratio, ...
                     trained_flg, leaned_flg);
       fprintf('epoch: %d, acc: %f, acccv: %f\n', ...
                                    epoch, acc, acccv);
       epoch_log(epoch) = epoch;
       acc_log(epoch) = acc;
       acccv_log(epoch) = acccv;
       epoch = epoch + 1;
    endif

    % �p�����[�^�̍X�V
    [W2, V_W2, H_W2] = optimizer(W2, dW2, V_W2, H_W2, iter, alpha, a1, a2);
    [b2, V_b2, H_b2] = optimizer(b2, db2, V_b2, H_b2, iter, alpha, a1, a2);
    [W3, V_W3, H_W3] = optimizer(W3, dW3, V_W3, H_W3, iter, alpha, a1, a2);
    [b3, V_b3, H_b3] = optimizer(b3, db3, V_b3, H_b3, iter, alpha, a1, a2);
    [W4, V_W4, H_W4] = optimizer(W4, dW4, V_W4, H_W4, iter, alpha, a1, a2);
    [b4, V_b4, H_b4] = optimizer(b4, db4, V_b4, H_b4, iter, alpha, a1, a2);
    [gamma2, V_gamma2, H_gamma2] = optimizer(gamma2, dgamma2, V_gamma2, H_gamma2, iter, alpha, a1, a2);
    [beta2, V_beta2, H_beta2] = optimizer(beta2, dbeta2, V_beta2, H_beta2, iter, alpha, a1, a2);
    [gamma3, V_gamma3, H_gamma3] = optimizer(gamma3, dgamma3, V_gamma3, H_gamma3, iter, alpha, a1, a2);
    [beta3, V_beta3, H_beta3] = optimizer(beta3, dbeta3, V_beta3, H_beta3, iter, alpha, a1, a2);
    
##    % �w�K��
##    if iter == 20
##       alpha = 0.05;
##    elseif iter == 50
##       alpha = 0.03;
##    endif
    
    % ���O
    iter_log(iter) = iter;
    J_log(iter) = J;
    
endfor


% �t���O
leaned_flg = 1;


##% ���z�m�F
##checkGradient();


##% �q�X�g�O����
##% iters_num=1�Ƃ��ĉB��w�̏o��(A2)�̕��z���m�F
##hist(A2(:));
##fprintf('histogram of A2. Press enter to continue.\n');
##pause;


##% �w�K�Ȑ�
##plot(iter_log, J_log);
##% �F�����x
##plot(epoch_log, acc_log, epoch_log, acccv_log);
##legend('train', 'cv');


##% �摜�`��E���_
##fprintf('predicting ...\n');
##Image = X(:,:,:,1:2); % 2�s�ڂ̓_�~�[
##Image(:,:,:,1) = imread('image_2.bmp');
##predict(Image, y(1:2), fp_actfunc, W2, b2, W3, b3, W4, b4, gamma2, beta2, gamma3, beta3, pad, stride, ph2, pw2, stride2, f1_2, f6_2, f1_3, f6_3, dropout_ratio, trained_flg, leaned_flg);


##% J���ʊ֐����m�F
##x1 = linspace(0.1, 1, 10); % x1=[0.1:0.1:1]�ł��������Ԋu�ɒ��ӁD�Ōオ1����Ȃ����Ƃ���
##x2 = linspace(0.1, 1, 10);
##[X1, X2] = meshgrid(x1, x2); % x1,x2�̑S�Ă̑g�����̍s��𐶐�
##Y1 = -( log(X1) + log(1-X2) );
##surf(X1, X2, Y1) % 3�����v���b�g


##% ���s���Ԍv��
##profile on;
##checkGradient()
##profile off;
##profile_data = profile ("info");
##profshow(profile_data, 10);
##tic;     % �v���J�n
##toc      % �o�ߎ��Ԃ��o��
##t = toc; % �o�ߎ��Ԃ��i�[
