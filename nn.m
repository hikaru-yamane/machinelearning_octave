% 変数をメモリから解放; すべてのフィギュア閉じる; コマンドウィンドウのクリア
clear; close all; clc;


% nn構成
% nを使うか保留
input_layer_size  = 400; % 20*20 input units
hidden_layer_size = 25;  % 25 hidden units
output_layer_size = 10;  % 10 output units(0は10とラベル)


% データ読込 X:5000*400(-0.2-1.2に正規化), y:5000*1(ベクトル，0は10とラベル)
% .txtだと読込遅すぎる
fprintf('data loading ...\n');
load ex4data1.mat;
% データ分割(6:2:2) →今はtest使わないから8:2
m = size(X, 1);
randInd = randperm(m); % 1:mをランダムに置き換え
Xcv = X(randInd(1:m*2/10), :);
ycv = y(randInd(1:m*2/10), :);
temp = m*2/10;
##Xtest = X(randInd(temp+1:(temp+1+m*2/10)-1), :);
##ytest = y(randInd(temp+1:(temp+1+m*2/10)-1), :);
##temp = temp + m*2/10;
X = X(randInd(temp+1:end), :);
y = y(randInd(temp+1:end), :);
% one-hot表現に変換
##Y = zeros(m, output_layer_size);
##for i = 1:output_layer_size
##    Y(:, i) = Y(:, i) + y==i;
##endfor
Y = y==[1:output_layer_size]; % m*10(one-shot表現，行列)
Ycv = ycv==[1:output_layer_size];
##Ytest = ytest==[1:output_layer_size];


% パラメータ(重みとバイアス)のランダム初期化
% Xavierの初期値(sigmoid, tanh)
##init_epsilon2 = sqrt(1/input_layer_size);
##init_epsilon3 = sqrt(1/hidden_layer_size);
##% Heの初期値(ReLU)
init_epsilon2 = sqrt(2/input_layer_size);
init_epsilon3 = sqrt(2/hidden_layer_size);
% ガウス分布で初期化
W2 = randn(input_layer_size, hidden_layer_size)*init_epsilon2;
b2 = zeros(1, hidden_layer_size);
W3 = randn(hidden_layer_size, output_layer_size)*init_epsilon3;
b3 = zeros(1, output_layer_size);
% パラメータ(正規化レイヤ)の初期化
gamma2 = ones(1, hidden_layer_size);
beta2 = zeros(1, hidden_layer_size);


% ハイパーパラメータ
alpha = 0.1;     % 学習率learning rate 0.01 AdaGrad
lambda = 1.0;    % 正則化の重みregularization weight
iters_num = 10; % 勾配降下法の繰り返し回数400


% 勾配降下法
fprintf('gradient descent ...\n');
for iter = 1:iters_num
    
    % 初期化
    if iter == 1
       % 関数
       fp_actfunc = @fp_ReLU;
       bp_actfunc = @bp_ReLU;
       fp_lastlayer = @fp_softmaxWithLoss;
       bp_lastlayer = @bp_softmaxWithLoss;
       % ログ
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
    
    % 認識精度
    [maxVal, maxInd] = max(Z3, [], 2); % 横に最大値
    accuracy = mean(double( maxInd==y ))*100;
    
    % Jcvを計算
    A1      = Xcv;
    Z2      = fp_affine(A1, W2, b2);
    A2      = fp_actfunc(Z2);
    Z3      = fp_affine(A2, W3, b3);
    [A3, Jcv] = fp_lastlayer(Z3, Ycv, W2, W3, 0, lambda);
    % acccvを計算
    [maxVal, maxInd] = max(Z3, [], 2); % 横に最大値
    acccv = mean(double( maxInd==ycv ))*100;

    % パラメータの更新
    W3 = W3 - alpha*dW3;
    b3 = b3 - alpha*db3;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    
    fprintf('Iter: %d, Cost: %f, Acc: %f\n' ...
                            , iter, J, accuracy);
    
    % ログ
    iter_log(iter) = iter;
    J_log(iter) = J;
    Jcv_log(iter) = Jcv;
    accuracy_log(iter) = accuracy;
    acccv_log(iter) = acccv;

endfor


##% 勾配確認
##checkGradient();


##% ヒストグラム
##% iters_num=1として隠れ層の出力(A2)の分布を確認
##hist(A2(:));
##fprintf('histogram of A2. Press enter to continue.\n');
##pause;


##% 学習曲線
##plot(iter_log, J_log, iter_log, Jcv_log);
##legend('train', 'cv');
##% 認識精度
##plot(iter_log, accuracy_log, iter_log, acccv_log);
##legend('train', 'cv');


##% 画像描画・推論
##fprintf('predicting ...\n');
##Image = imread('image_2.bmp');
##predict(Image, Y, W2, b2, W3, b3, lambda);


##% Jが凸関数か確認
##x1 = linspace(0.1, 1, 10); % x1=[0.1:0.1:1]でもいいが間隔に注意．最後が1じゃないことあり
##x2 = linspace(0.1, 1, 10);
##[X1, X2] = meshgrid(x1, x2); % x1,x2の全ての組合せの行列を生成
##Y1 = -( log(X1) + log(1-X2) );
##surf(X1, X2, Y1) % 3次元プロット


##% 実行時間計測
##profile on;
##checkGradient()
##profile off;
##profile_data = profile ("info");
##profshow(profile_data, 10);
##tic;     % 計測開始
##toc      % 経過時間を出力
##t = toc; % 経過時間を格納
