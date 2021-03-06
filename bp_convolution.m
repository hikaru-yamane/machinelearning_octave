function [dIn, dW, db] = bp_convolution(dOut, In, In_two, W, W_two, pad=0, stride=1, lambda)
  
  % 初期化
  [h, w, c, m] = size(In); % octaveはこの順番
  [fh, fw, c, fm] = size(W);
  out_h = 1 + (h + 2*pad - fh)/stride; % 整数になるよう設定
  out_w = 1 + (w + 2*pad - fw)/stride; % 整数になるよう設定
  
  % four2two
  dOut = permute(dOut, [1 2 4 3]); % (out_h,out_w,m,fm)
  dOut_two = reshape(dOut, [m*out_w*out_h, fm]);
  
  % bp_affine
  dIn_two = dOut_two*W_two'; % (m*out_w*out_h, c*fw*fh)
  dW_two = In_two'*dOut_two; % (c*fw*fh, fm)
  db = sum(dOut_two, 1);     % 縦に足す
  
  % two2four
##  dIn = zeros(h, w, c, m);
##  cnt = 1;
##  for j = 1:out_w
##  for i = 1:out_h
##      Temp = dIn_two(cnt:out_w*out_h:m*out_w*out_h, :); % (m, c*fw*fh)
##      Temp = reshape(Temp', [fh, fw, c, m]); % (fh, fw, c, m)
##      dIn(i:i+fh-1, j:j+fw-1, :, :) += Temp; % (fh, fw, c, m) % +=に注意
##      cnt = cnt + 1;
##  endfor
##  endfor
  dIn_six = reshape(dIn_two, [out_h,out_w,m,fh,fw,c]);
  dIn_six = permute(dIn_six, [4 5 1 2 6 3]); % (fh,fw,out_h,out_w,c,m)
  dIn = zeros(h, w, c, m);
  for i = 1:out_h
  for j = 1:out_w
      Temp = dIn_six(:,:,i,j,:,:); % (fh, fw, 1, 1, c, m)
      Temp = reshape(Temp, [fh, fw, c, m]);
      dIn(i:i+fh-1, j:j+fw-1, :, :) += Temp; % +=に注意
  endfor
  endfor
  dW = reshape(dW_two, [fh, fw, c, fm]);
  
  % 正則化
  dW = dW + (lambda/m)*W;
  
endfunction