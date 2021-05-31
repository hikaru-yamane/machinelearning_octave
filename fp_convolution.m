function [Out, In_two, W_two] = fp_convolution(In, W, b, pad=0, stride=1)
  
  % ‰Šú‰»
  [h, w, c, m] = size(In); % octave‚Í‚±‚Ì‡”Ô
  [fh, fw, c, fm] = size(W);
  out_h = 1 + (h + 2*pad - fh)/stride; % ®”‚É‚È‚é‚æ‚¤İ’è
  out_w = 1 + (w + 2*pad - fw)/stride; % ®”‚É‚È‚é‚æ‚¤İ’è
  
  % four2two
  % 6ŸŒ³g‚Á‚½‚Ù‚¤‚ª‘¬‚¢(m*out_w*out_h, c*fw*fh)¨(out_h,out_w,m,fh,fw,c)¨(fh,fw,out_h,out_w,c,m)
##  In_two = zeros(m*out_w*out_h, c*fw*fh);
##  cnt = 1;
##  for j = 1:out_w
##  for i = 1:out_h
##      Temp = In(i:i+fh-1, j:j+fw-1, :, :); % (fh, fw, c, m)
##      Temp = reshape(Temp, [c*fw*fh, m])'; % (m, c*fw*fh)
##      In_two(cnt:out_w*out_h:m*out_w*out_h, :) = Temp; % (m*out_w*out_h, c*fw*fh)
##      cnt = cnt + 1;
##  endfor
##  endfor
  In_six = zeros(fh,fw,out_h,out_w,c,m);
  for i = 1:out_h
  for j = 1:out_w
      In_six(:,:,i,j,:,:) = In(i:i+fh-1, j:j+fw-1, :, :); % (fh, fw, c, m)
  endfor
  endfor
  In_six = permute(In_six, [3 4 6 1 2 5]); % (out_h,out_w,m,fh,fw,c)
  In_two = reshape(In_six, [m*out_w*out_h, c*fw*fh]);
  W_two = reshape(W, [c*fw*fh, fm]);
  
  % fp_affine
  Out_two = In_two*W_two + b; % (m*out_w*out_h, fm)
  
  % two2four
  Out = reshape(Out_two, [out_h,out_w,m,fm]);
  Out = permute(Out, [1 2 4 3]); % (out_h,out_w,fm,m)
  
endfunction