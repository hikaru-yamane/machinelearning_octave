function [Out, J] = fp_softmaxWithLoss(In, Y, W2, W3, W4, lambda)
  % debug���CY�̏����ɒ���
  
  % ������
  m = size(In, 1);
  epsilon = 1e-8; % �ΐ��̃I�[�o�[�t���[�΍�
  
  % �w���̃I�[�o�[�t���[�΍�
  X = In - max(In,[],2);
  
  Out = exp(X)./sum(exp(X),2) + epsilon; % ���ɑ���
  J = (-1/m)*sum(sum( Y.*log(Out) ));
  
  % ������
  J = J + (lambda/(2*m))*sum(sum(sum(sum( W2.^2 )))) ...
        + (lambda/(2*m))*sum(sum( W3.^2 )) ...
        + (lambda/(2*m))*sum(sum( W4.^2 ));
  
endfunction