% 0-9�菑�������F��
% Ctrl+R/Ctrl+Shift+R: �R�����g�A�E�g/�A���R�����g
% Tab/Ctrl+Shift+Tab: ������/��������߂�
% F10: �X�e�b�v���s
% ���K���C�n�C�p�[�̍œK���Ccv��test�����CJ��acc�̃O���t�C�B��w����
% �{�Ɠ��������CReLU��softmax
% ����摜�����z�m�F���{�Ɠ�������


% �ϐ���������������; ���ׂẴt�B�M���A����; �R�}���h�E�B���h�E�̃N���A
clear; close all; clc;


% nn�\��
% n���g�����ۗ�
input_layer_size  = 400; % 20*20 input units
hidden_layer_size = 25;  % 25 hidden units
output_layer_size = 10;  % 10 output units(0��10�ƃ��x��)


% �f�[�^�Ǎ� X:5000*400(-0.2-1.2�ɐ��K��), y:5000*1(�x�N�g���C0��10�ƃ��x��)
% .txt���ƓǍ��x������
fprintf('loading ...\n');
load ex4data1.mat;
m = size(X, 1);
% one-shot�\���ɕϊ�
##Y = zeros(m, output_layer_size);
##for i = 1:output_layer_size
##    Y(:, i) = Y(:, i) + y==i;
##endfor
Y = y==[1:output_layer_size]; % m*10(one-shot�\���C�s��)
##% X�̐��K�� ��sigma�Ƀ[�����ł��邩�璆�~
##% �eZ�̐��K���͕ۗ� BP�̌v�Z���ς��̂Œ���
##mu = mean(X);        % 1*400 ����
##sigma = std(X);      % 1*400 �W���΍�
##X = (X - mu)./sigma; % ./����Ȃ��ƃu���[�h�L���X�g��肭�����Ȃ�


% �p�����[�^(�d�݂ƃo�C�A�X)�̃����_��������
##% Theta1: 25*401, Theta2: 10*26
##load ex4weights.mat;
##W2 = Theta1(:, 2:end)'; % 400*25
##b2 = Theta1(:, 1)';     % 1*25
##W3 = Theta2(:, 2:end)'; % 25*10
##b3 = Theta2(:, 1)';     % 1*10
% Xavier�̏����l(sigmoid, tanh)
init_epsilon2 = sqrt(1/input_layer_size);
init_epsilon3 = sqrt(1/hidden_layer_size);
##% He�̏����l(ReLU)
##init_epsilon2 = sqrt(2/input_layer_size);
##init_epsilon3 = sqrt(2/hidden_layer_size);
% �K�E�X���z�ŏ�����
W2 = randn(input_layer_size, hidden_layer_size)*init_epsilon2;
b2 = randn(1, hidden_layer_size)*init_epsilon2;
W3 = randn(hidden_layer_size, output_layer_size)*init_epsilon3;
b3 = randn(1, output_layer_size)*init_epsilon3;


% �n�C�p�[�p�����[�^
alpha = 1;       % �w�K��learning rate 0.01 AdaGrad
lambda = 1;      % �������̏d��regularization weight
iters_num = 400; % ���z�~���@�̌J��Ԃ���400


% ���z�~���@
% ��K�͂�nn����������Ƃ��͋K�����ƒT����(�ϐ����P�������₷)�C�{�Ɠ��l�ɂP�X�e�b�v���ƂɊ֐������(�֐����܂Ƃ߂�N���X)
fprintf('gradient descent ...\n');
for iter = 1:iters_num
    
    % ������
    if iter == 1
       iter_log = zeros(iters_num, 1);
       J_log = zeros(iters_num, 1);
       accuracy_log = zeros(iters_num, 1);
    endif

    % FP
    % Z2��Z3�͏ȗ��ł��邪�ŏ������珑���Ƃ�
    A1 = X;           % m*400
    Z2 = A1*W2 + b2;  % m*25
    A2 = sigmoid(Z2); % m*25
    Z3 = A2*W3 + b3;  % m*10
    A3 = sigmoid(Z3); % m*10

    ##J = 0;
    ##for i = 1:m
    ##for j = 1:output_layer_size
    ##    J = J + (-1/m)*( Y(i, j)*log(A3(i, j)) + (1 - Y(i, j))*log(1 - A3(i, j)) );
    ##endfor
    ##endfor
    J = (-1/m)*sum(sum( Y.*log(A3) + (1 - Y).*log(1 - A3) ));
    % ������
    J = J + (lambda/(2*m))*sum(sum( W2.^2 )) ...
          + (lambda/(2*m))*sum(sum( W3.^2 ));

    % BP
    dLdZ3 = A3 - Y;                    % m*10
    dW3 = (1/m)*A2'*dLdZ3;             % 25*10
    db3 = (1/m)*sum(dLdZ3, 1);         % 1*10(�c�ɑ���)
    dLdZ2 = (dLdZ3*W3').*(A2.*(1-A2)); % m*25
    dW2 = (1/m)*X'*dLdZ2;              % 400*25
    db2 = (1/m)*sum(dLdZ2, 1);         % 1*25(�c�ɑ���)
    % ������
    dW3 = dW3 + (lambda/m)*W3;
    dW2 = dW2 + (lambda/m)*W2;

    % �p�����[�^�̍X�V
    W3 = W3 - alpha*dW3;
    b3 = b3 - alpha*db3;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    
    % �F�����x
    [maxVal, maxInd] = max(A3, [], 2);
    accuracy = mean(double( maxInd==y ))*100;
    
    fprintf('Iteration: %d, Cost: %f, Accuracy: %f\n' ...
                                 , iter, J, accuracy);
    
    % ���O
    iter_log(iter) = iter;
    J_log(iter) = J;
    accuracy_log(iter) = accuracy;

endfor


##% ���z�m�F
##fprintf('gradient checking ...\n');
##X = rand(1, 2); % ���f�[�^���ƃ[������������
##Y = rand(1, 2);
##m = size(X, 1);
##W2 = randn(2, 2);
##b2 = randn(1, 2);
##W3 = randn(2, 2);
##b3 = randn(1, 2);
##% ��͔���
##A1 = X;           % m*400
##Z2 = A1*W2 + b2;  % m*25
##A2 = sigmoid(Z2); % m*25
##Z3 = A2*W3 + b3;  % m*10
##A3 = sigmoid(Z3); % m*10
##J = (-1/m)*sum(sum( Y.*log(A3) + (1 - Y).*log(1 - A3) ));
##dLdZ3 = A3 - Y;                    % m*10
##dW3 = (1/m)*A2'*dLdZ3;             % 25*10
##db3 = (1/m)*sum(dLdZ3, 1);         % 1*10(�c�ɑ���)
##dLdZ2 = (dLdZ3*W3').*(A2.*(1-A2)); % m*25
##dW2 = (1/m)*X'*dLdZ2;              % 400*25
##db2 = (1/m)*sum(dLdZ2, 1);         % 1*25(�c�ɑ���)
##dparams = [dW2(:); db2(:); dW3(:); db3(:)];
##% ���l����
##epsilon = 1e-4; % e�͑g�ݍ��ݕϐ�
##params_numerical = [W2(:); b2(:); W3(:); b3(:)]; % �u:�v: 2��ڂ�1��ڂ̉�
##dparams_numerical = zeros(size(params_numerical));
##for i = 1:length(params_numerical)
##    temp = params_numerical(i);
##    params_numerical(i) = temp + epsilon;
##    W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: �u:�v�̋t�̑��� numel: �v�f�̌��̑��a
##    b2 = reshape(params_numerical(numel(W2)+1:numel(W2)+1+numel(b2)-1), size(b2));
##    W3 = reshape(params_numerical(numel(W2)+numel(b2)+1:numel(W2)+numel(b2)+1+numel(W3)-1), size(W3));
##    b3 = reshape(params_numerical(numel(W2)+numel(b2)+numel(W3)+1:end), size(b3));
##    A1 = X;           % m*400
##    Z2 = A1*W2 + b2;  % m*25
##    A2 = sigmoid(Z2); % m*25
##    Z3 = A2*W3 + b3;  % m*10
##    A3 = sigmoid(Z3); % m*10
##    J_plus = (-1/m)*sum(sum( Y.*log(A3) + (1 - Y).*log(1 - A3) ));
##    params_numerical(i) = temp - epsilon;
##    W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: �u:�v�̋t�̑��� numel: �v�f�̌��̑��a
##    b2 = reshape(params_numerical(numel(W2)+1:numel(W2)+1+numel(b2)-1), size(b2));
##    W3 = reshape(params_numerical(numel(W2)+numel(b2)+1:numel(W2)+numel(b2)+1+numel(W3)-1), size(W3));
##    b3 = reshape(params_numerical(numel(W2)+numel(b2)+numel(W3)+1:end), size(b3));
##    A1 = X;           % m*400
##    Z2 = A1*W2 + b2;  % m*25
##    A2 = sigmoid(Z2); % m*25
##    Z3 = A2*W3 + b3;  % m*10
##    A3 = sigmoid(Z3); % m*10
##    J_minus = (-1/m)*sum(sum( Y.*log(A3) + (1 - Y).*log(1 - A3) ));
##    dparams_numerical(i) = (J_plus - J_minus)/(2*epsilon);
##    params_numerical(i) = temp;
##endfor
##% ��͔����Ɛ��l�����̍�
##disp([dparams_numerical dparams]);
##gap = norm(dparams_numerical - dparams) ...
##     /norm(dparams_numerical + dparams);
##fprintf('gap(less than 1e-9): %g\n', gap); % �u%g�v: %f���������̌���\��


##% J���ʊ֐����m�F
##x1 = linspace(0.1, 1, 10); % x1=[0.1:0.1:1]�ł��������Ԋu�ɒ��ӁD�Ōオ1����Ȃ����Ƃ���
##x2 = linspace(0.1, 1, 10);
##[X1, X2] = meshgrid(x1, x2); % x1,x2�̑S�Ă̑g�����̍s��𐶐�
##Y1 = -( log(X1) + log(1-X2) );
##surf(X1, X2, Y1) % 3�����v���b�g


##% �q�X�g�O����
##hist(A2(:));
##fprintf('histogram of A2. Press enter to continue.\n');
##pause;
##hist(A3(:));
##fprintf('histogram of A3. Press enter to continue.\n');
##pause;


##% �w�K�Ȑ�
##plot(iter_log, J_log);


% �摜�`��E���_
fprintf('estimating ...\n');
Image = imread('image_2.bmp');
##imagesc(reshape(X(1,:), [20, 20])); colorbar;
imagesc(Image); colorbar;
X = reshape(Image, [1, 400]);
A1 = X;           % m*400
Z2 = A1*W2 + b2;  % m*25
A2 = sigmoid(Z2); % m*25
Z3 = A2*W3 + b3;  % m*10
A3 = sigmoid(Z3); % m*10
[maxVal, maxInd] = max(A3, [], 2); % ���ɍő�l�C�C���f�b�N�X
fprintf('estimated number: %d\n', maxInd);


% �B��w�̉���


##% �F�����x
##plot(iter_log, accuracy_log);
