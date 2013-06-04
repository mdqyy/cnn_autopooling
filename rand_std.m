function out = rand_std(w, h, numinp) 
%rand_std train convolutional neural network using stochastic Levenberg-Marquardt  
% Generates small random nums centred at 0.
%
%  Syntax
%  
%    [cnet, perf_plot] = train(cnet,Ip,labels)
%    
%  Description
%   Input:
%    w - kernel height (of the local receptive field of the conv layer)
%    h - kernel width (of the local receptive field of the conv layer)
%    numinp - number of weights that need to be trained; it is equal to the
%    kernel width x kernel height x number of kernels (e.g. for a 5x5
%    receptive field and 6 kernels, the num of weights is 25x6=150)
%
%   Output:
%    out - w x h dimensional matrix of random doubles
%
%(c) Sirotenko Mikhail, 2009
%Генерирует матрицу случайных чисел в диапазоне от -1 до 1
%numinp - количество входов в нейрон. Используется для выполнения правила
%sigma=m^-1/2, где сигма - среднеквадратич. отклонение, а m - количество
%весов входящих в нейрон
  sigma = numinp^(-1/2); % 1/sqrt(num_weights) => 1x1 double
  out = (rand(w,h) - ones(w,h)/2); % random nums in the interval (-0.5, 0.5)

  if(w*h>1)
    outstd = mean(std(reshape(out,1,[])));      

  else
      outstd=1;
  end
  out = out*sigma/outstd;
