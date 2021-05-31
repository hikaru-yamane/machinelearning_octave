function [Out, Mask] = fp_pooling(In, ph, pw, pad=0, stride)
  
  % ������
  [h, w, c, m] = size(In); % octave�͂��̏���
  out_h = 1 + (h - ph)/stride; % �����ɂȂ�悤�ݒ�
  out_w = 1 + (w - pw)/stride; % �����ɂȂ�悤�ݒ�
  
  % four2two
  % (m*c*out_w*out_h, pw*ph)��(out_h,out_w,c,m,ph,pw)��(ph,pw,out_h,out_w,c,m)
  In_six = zeros(ph,pw,out_h,out_w,c,m);
  for i = 1:out_h % �o�̗͂v�f�̓Y��
  for j = 1:out_w
      start_i = (i-1)*stride+1; % ���̗͂v�f�̓Y��
      start_j = (j-1)*stride+1;
      In_six(:,:,i,j,:,:) = In(start_i:start_i+ph-1, start_j:start_j+pw-1, :, :); % (ph, pw, c, m)
  endfor
  endfor
  In_six = permute(In_six, [3 4 5 6 1 2]); % (out_h,out_w,c,m,ph,pw)
  In_two = reshape(In_six, [m*c*out_w*out_h, pw*ph]);
  
  % fp_maxPooling
  Out_two = max(In_two,[],2); % �� % (m*c*out_w*out_h, 1)
  Mask = In_two==Out_two;
  
  % two2four
  Out = reshape(Out_two, [out_h,out_w,c,m]);
  
endfunction