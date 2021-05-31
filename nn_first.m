% 0-9手書き数字認識
% Ctrl+R/Ctrl+Shift+R: コメントアウト/アンコメント
% Tab/Ctrl+Shift+Tab: 字下げ/字下げを戻す
% F10: ステップ実行
% 正規化，ハイパーの最適化，cvとtest分離，Jとaccのグラフ，隠れ層可視化
% 本と同じ実装，ReLUとsoftmax
% 自作画像→勾配確認→本と同じ実装


% 変数をメモリから解放; すべてのフィギュア閉じる; コマンドウィンドウのクリア
clear; close all; clc;


% nn構成
% nを使うか保留
input_layer_size  = 400; % 20*20 input units
hidden_layer_size = 25;  % 25 hidden units
output_layer_size = 10;  % 10 output units(0は10とラベル)


% データ読込 X:5000*400(-0.2-1.2に正規化), y:5000*1(ベクトル，0は10とラベル)
% .txtだと読込遅すぎる
fprintf('loading ...\n');
load ex4data1.mat;
m = size(X, 1);
% one-shot表現に変換
##Y = zeros(m, output_layer_size);
##for i = 1:output_layer_size
##    Y(:, i) = Y(:, i) + y==i;
##endfor
Y = y==[1:output_layer_size]; % m*10(one-shot表現，行列)
##% Xの正規化 ←sigmaにゼロができるから中止
##% 各Zの正規化は保留 BPの計算式変わるので注意
##mu = mean(X);        % 1*400 平均
##sigma = std(X);      % 1*400 標準偏差
##X = (X - mu)./sigma; % ./じゃないとブロードキャスト上手くいかない


% パラメータ(重みとバイアス)のランダム初期化
##% Theta1: 25*401, Theta2: 10*26
##load ex4weights.mat;
##W2 = Theta1(:, 2:end)'; % 400*25
##b2 = Theta1(:, 1)';     % 1*25
##W3 = Theta2(:, 2:end)'; % 25*10
##b3 = Theta2(:, 1)';     % 1*10
% Xavierの初期値(sigmoid, tanh)
init_epsilon2 = sqrt(1/input_layer_size);
init_epsilon3 = sqrt(1/hidden_layer_size);
##% Heの初期値(ReLU)
##init_epsilon2 = sqrt(2/input_layer_size);
##init_epsilon3 = sqrt(2/hidden_layer_size);
% ガウス分布で初期化
W2 = randn(input_layer_size, hidden_layer_size)*init_epsilon2;
b2 = randn(1, hidden_layer_size)*init_epsilon2;
W3 = randn(hidden_layer_size, output_layer_size)*init_epsilon3;
b3 = randn(1, output_layer_size)*init_epsilon3;


% ハイパーパラメータ
alpha = 1;       % 学習率learning rate 0.01 AdaGrad
lambda = 1;      % 正則化の重みregularization weight
iters_num = 400; % 勾配降下法の繰り返し回数400


% 勾配降下法
% 大規模なnnを実装するときは規則性と探すか(変数を１次元増やす)，本と同様に１ステップごとに関数を作る(関数をまとめるクラス)
fprintf('gradient descent ...\n');
for iter = 1:iters_num
    
    % 初期化
    if iter == 1
       iter_log = zeros(iters_num, 1);
       J_log = zeros(iters_num, 1);
       accuracy_log = zeros(iters_num, 1);
    endif

    % FP
    % Z2とZ3は省略できるが最初だから書いとく
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
    % 正則化
    J = J + (lambda/(2*m))*sum(sum( W2.^2 )) ...
          + (lambda/(2*m))*sum(sum( W3.^2 ));

    % BP
    dLdZ3 = A3 - Y;                    % m*10
    dW3 = (1/m)*A2'*dLdZ3;             % 25*10
    db3 = (1/m)*sum(dLdZ3, 1);         % 1*10(縦に足す)
    dLdZ2 = (dLdZ3*W3').*(A2.*(1-A2)); % m*25
    dW2 = (1/m)*X'*dLdZ2;              % 400*25
    db2 = (1/m)*sum(dLdZ2, 1);         % 1*25(縦に足す)
    % 正則化
    dW3 = dW3 + (lambda/m)*W3;
    dW2 = dW2 + (lambda/m)*W2;

    % パラメータの更新
    W3 = W3 - alpha*dW3;
    b3 = b3 - alpha*db3;
    W2 = W2 - alpha*dW2;
    b2 = b2 - alpha*db2;
    
    % 認識精度
    [maxVal, maxInd] = max(A3, [], 2);
    accuracy = mean(double( maxInd==y ))*100;
    
    fprintf('Iteration: %d, Cost: %f, Accuracy: %f\n' ...
                                 , iter, J, accuracy);
    
    % ログ
    iter_log(iter) = iter;
    J_log(iter) = J;
    accuracy_log(iter) = accuracy;

endfor


##% 勾配確認
##fprintf('gradient checking ...\n');
##X = rand(1, 2); % 生データだとゼロが多すぎる
##Y = rand(1, 2);
##m = size(X, 1);
##W2 = randn(2, 2);
##b2 = randn(1, 2);
##W3 = randn(2, 2);
##b3 = randn(1, 2);
##% 解析微分
##A1 = X;           % m*400
##Z2 = A1*W2 + b2;  % m*25
##A2 = sigmoid(Z2); % m*25
##Z3 = A2*W3 + b3;  % m*10
##A3 = sigmoid(Z3); % m*10
##J = (-1/m)*sum(sum( Y.*log(A3) + (1 - Y).*log(1 - A3) ));
##dLdZ3 = A3 - Y;                    % m*10
##dW3 = (1/m)*A2'*dLdZ3;             % 25*10
##db3 = (1/m)*sum(dLdZ3, 1);         % 1*10(縦に足す)
##dLdZ2 = (dLdZ3*W3').*(A2.*(1-A2)); % m*25
##dW2 = (1/m)*X'*dLdZ2;              % 400*25
##db2 = (1/m)*sum(dLdZ2, 1);         % 1*25(縦に足す)
##dparams = [dW2(:); db2(:); dW3(:); db3(:)];
##% 数値微分
##epsilon = 1e-4; % eは組み込み変数
##params_numerical = [W2(:); b2(:); W3(:); b3(:)]; % 「:」: 2列目を1列目の下
##dparams_numerical = zeros(size(params_numerical));
##for i = 1:length(params_numerical)
##    temp = params_numerical(i);
##    params_numerical(i) = temp + epsilon;
##    W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: 「:」の逆の操作 numel: 要素の個数の総和
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
##    W2 = reshape(params_numerical(1:numel(W2)), size(W2)); % reshape: 「:」の逆の操作 numel: 要素の個数の総和
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
##% 解析微分と数値微分の差
##disp([dparams_numerical dparams]);
##gap = norm(dparams_numerical - dparams) ...
##     /norm(dparams_numerical + dparams);
##fprintf('gap(less than 1e-9): %g\n', gap); % 「%g」: %fよりも多くの桁を表示


##% Jが凸関数か確認
##x1 = linspace(0.1, 1, 10); % x1=[0.1:0.1:1]でもいいが間隔に注意．最後が1じゃないことあり
##x2 = linspace(0.1, 1, 10);
##[X1, X2] = meshgrid(x1, x2); % x1,x2の全ての組合せの行列を生成
##Y1 = -( log(X1) + log(1-X2) );
##surf(X1, X2, Y1) % 3次元プロット


##% ヒストグラム
##hist(A2(:));
##fprintf('histogram of A2. Press enter to continue.\n');
##pause;
##hist(A3(:));
##fprintf('histogram of A3. Press enter to continue.\n');
##pause;


##% 学習曲線
##plot(iter_log, J_log);


% 画像描画・推論
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
[maxVal, maxInd] = max(A3, [], 2); % 横に最大値，インデックス
fprintf('estimated number: %d\n', maxInd);


% 隠れ層の可視化


##% 認識精度
##plot(iter_log, accuracy_log);
