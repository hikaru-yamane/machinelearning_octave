function dIn = bp_pooling(dOut, In, Mask, ph, pw, pad=0, stride)
  
  % ‰Šú‰»
  [h, w, c, m] = size(In); % octave‚Í‚±‚Ì‡”Ô
  out_h = 1 + (h - ph)/stride; % ®”‚É‚È‚é‚æ‚¤İ’è
  out_w = 1 + (w - pw)/stride; % ®”‚É‚È‚é‚æ‚¤İ’è
  
  % four2two
  dOut_two = reshape(dOut, [m*c*out_w*out_h, 1]);
  
  % bp_maxPooling
  dIn_two = Mask.*dOut_two; % (m*c*out_w*out_h, pw*ph)
  
  % two2four
  dIn_six = reshape(dIn_two, [out_h,out_w,c,m,ph,pw]);
  dIn_six = permute(dIn_six, [5 6 1 2 3 4]); % (ph,pw,out_h,out_w,c,m)
  dIn = zeros(h, w, c, m);
  for i = 1:out_h
  for j = 1:out_w
      Temp = dIn_six(:,:,i,j,:,:); % (ph, pw, 1, 1, c, m)
      Temp = reshape(Temp, [ph, pw, c, m]);
      start_i = (i-1)*stride+1;
      start_j = (j-1)*stride+1;
      dIn(start_i:start_i+ph-1, start_j:start_j+pw-1, :, :) += Temp; % +=‚É’ˆÓ
  endfor
  endfor
  
endfunction