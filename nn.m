% �ϐ���������������; ���ׂẴt�B�M���A����; �R�}���h�E�B���h�E�̃N���A
clear; close all; clc;


% nn�\��
% n���g�����ۗ�
input_layer_size  = 400; % 20*20 input units
hidden_layer_size = 25;  % 25 hidden units
output_layer_size = 10;  % 10 output units(0��10�ƃ��x��)


% �f�[�^�Ǎ� X:5000*400(-0.2-1.2�ɐ��K��), y:5000*1(�x�N�g���C0��10�ƃ��x��)
% .txt���ƓǍ��x������
fprintf('data loading ...\n');
load ex4data1.mat;
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
Y = y==[1:output_layer_size]; % m*10(one-shot�\���C�s��)
Ycv = ycv==[1:output_layer_size];
##Ytest = ytest==[1:output_layer_size];


% �p�����[�^(�d�݂ƃo�C�A�X)�̃����_��������
% Xavier�̏����l(sigmoid, tanh)
##init_epsilon2 = sqrt(1/input_layer_size);
##init_epsilon3 = sqrt(1/hidden_layer_size);
##% He�̏����l(ReLU)
init_epsilon2 = sqrt(2/input_layer_size);
init_epsilon3 = sqrt(2/hidden_layer_size);
% �K�E�X���z�ŏ�����
W2 = randn(input_layer_size, hidden_layer_size)*init_epsilon2;
b2 = zeros(1, hidden_layer_size);
W3 = randn(hidden_layer_size, output_layer_size)*init_epsilon3;
b3 = zeros(1, output_layer_size);
% �p�����[�^(���K�����C��)�̏�����
gamma2 = ones(1, hidden_layer_size);
beta2 = zeros(1, hidden_layer_size);


% �n�C�p�[�p�����[�^
alpha = 0.1;     % �w�K��learning rate 0.01 AdaGrad
lambda = 1.0;    % �������̏d��regularization weight
iters_num = 10; % ���z�~���@�̌J��Ԃ���400


% ���z�~���@
fprintf('gradient descent ...\n');
for iter = 1:iters_num
    
    % ������
    if iter == 1
       % �֐�
       fp_actfunc = @fp_ReLU;
       bp_actfunc = @bp_ReLU;
       fp_lastlayer = @fp_softmaxWithLoss;
       bp_lastlayer = @bp_softmaxWithLoss;
       % ���O
       iter_log = zeros(iters_num, 1);
       J_log = zeros(iters_num, 1);
       Jcv_log = zeros(iters_num, 1);
       accuracy_log = zeros(iters_num, 1);
       acccv_log = zeros(iters_num, 1);
    endif

    % FP
    A1      = X;
    Z2      = fp_affine(A1, W2, b2);
    A2      = fp_actfunc(Z2);
    Z3      = fp_affine(A2, W3, b3);
    [A3, J] = fp_lastlayer(Z3, Y, W2, W3, 0,lambda);

    % BP
    dZ3             = bp_lastlayer(dJ=1, A3, Y);
    [dA2, dW3, db3] = bp_affine(dZ3, A2, W3, lambda);
    dZ2             = bp_actfunc(dA2, Z2, A2);
    [dA1, dW2, db2] = bp_affine(dZ2, A1, W2, lambda);
    
    % �F�����x
    [maxVal, maxInd] = max(Z3, [], 2); % ���ɍő�l
    accuracy = mean(double( maxInd==y ))*100;
    
    % Jcv���v�Z
    A1      = Xcv;
    Z2      = fp_affine(A1, W2, b2);
    A2      = fp_actfunc(Z2);
    Z3      = fp_affine(A2, W3, b3);
    [A3, Jcv] = fp_lastlayer(Z3, Ycv, W2, W3, 0, lambda);
    % acccv���v�Z
    [maxVal, maxInd] = max(Z3, [], 2); % ���ɍő�l
    acccv = mean(double( maxInd==ycv ))*100;

    % �p�����[�^�̍X�V
    W3 = W3 - alpha*dW3;
    b3 = b3 - alpha*db3;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    
    fprintf('Iter: %d, Cost: %f, Acc: %f\n' ...
                            , iter, J, accuracy);
    
    % ���O
    iter_log(iter) = iter;
    J_log(iter) = J;
    Jcv_log(iter) = Jcv;
    accuracy_log(iter) = accuracy;
    acccv_log(iter) = acccv;

endfor


##% ���z�m�F
##checkGradient();


##% �q�X�g�O����
##% iters_num=1�Ƃ��ĉB��w�̏o��(A2)�̕��z���m�F
##hist(A2(:));
##fprintf('histogram of A2. Press enter to continue.\n');
##pause;


##% �w�K�Ȑ�
##plot(iter_log, J_log, iter_log, Jcv_log);
##legend('train', 'cv');
##% �F�����x
##plot(iter_log, accuracy_log, iter_log, acccv_log);
##legend('train', 'cv');


##% �摜�`��E���_
##fprintf('predicting ...\n');
##Image = imread('image_2.bmp');
##predict(Image, Y, W2, b2, W3, b3, lambda);


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
