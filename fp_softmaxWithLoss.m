function [Out, J] = fp_softmaxWithLoss(In, Y, W2, W3, W4, lambda)
  % debug時，Yの条件に注意
  
  % 初期化
  m = size(In, 1);
  epsilon = 1e-8; % 対数のオーバーフロー対策
  
  % 指数のオーバーフロー対策
  X = In - max(In,[],2);
  
  Out = exp(X)./sum(exp(X),2) + epsilon; % 横に足す
  J = (-1/m)*sum(sum( Y.*log(Out) ));
  
  % 正則化
  J = J + (lambda/(2*m))*sum(sum(sum(sum( W2.^2 )))) ...
        + (lambda/(2*m))*sum(sum( W3.^2 )) ...
        + (lambda/(2*m))*sum(sum( W4.^2 ));
  
endfunction