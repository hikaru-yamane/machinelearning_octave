function debugOptimizer(In)
  
  % ‰Šú‰»
  W = In;
  alpha = 1.01; % ŠwK—¦
  iters_num = 50;
  V = 0;
  H = 0;
  a1 = 0.9;
  a2 = 0.999;
  
  % ŠwK
  for iter = 1:iters_num
      
      if iter == 1
         iter_log = zeros(iters_num, 1);
         J_log = zeros(iters_num, 1);
      endif
      
      J = sum(sum(W.^2));
      dW = 2*W;
      
      [W, V, H] = opt_gd(W, dW, V, H, iter, alpha, a1, a2);     
      
      fprintf('Iter: %d, Cost: %f, |W|: %f\n' , iter, J, norm(W(:)));
      
      % ƒƒO
      iter_log(iter) = iter;
      J_log(iter) = J;
      
  endfor
  
  plot(iter_log, J_log);
  
endfunction
