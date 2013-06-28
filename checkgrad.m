function [diff,numgrad,grad] = checkgrad(cnet,num,Ip,d,order,numToCheck,ind,je)
%checkgrad: calculates a secant approximation to the error gradient, and
%compares it to the exact computed value of the gradient, to ascertain if
%the gradient computation is OK. Returns "diff", the difference between the
%approximated and computed gradient - a small value (< 10^-4) indicates the
%gradient is OK; otherwise there may be an error in the gradient
%computation.
%
%  Syntax
%  
%    [diff,numgrad,grad] = checkgrad(cnet,num,Ip,d,order,numToCheck,ind,je) 
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object, assuming it has been
%    simulated for the provided datapoint
%    num - number of parameters (weights and biases) in single-column weight
%    vector (which have to be learned), i.e. the size of the CNN
%    Ip - 1 x N cell array of M x M double matrices - the input examples
%    d - desired output, N x 1 double vector (one label for each example)
%    order - 1 means gradient, 2 - Hessian
%    numToCheck - 1 x 1 int, the number of gradients to check (since
%    checking all gradients in the network takes too long for deep nets)
%    ind - 1 x N int, the indices of the gradients to be checked
%    numerically (e.g., ind = [1] means check dEdW (wrt w1) for last layer)
%    je - num x 1 double vector of computed gradients
%    
%   Output:
%    diff - 1 x numToCheck double vector. The ratios of the norms of the 
%    difference and sum of the approximated gradients and the computed 
%    gradients, for each of the gradient indices that were requested to be
%    checked (as provided by "ind"). A small value (e.g. < 10^-4) indicates
%    that the computed gradient is correct.
%    
%    numgrad - numToCheck x 1 double vector. Gradients for all parameters, 
%    calculated using finite differences. If calcje works correctly, 
%    the numerical gradients should be almost the same as the gradients 
%    produced by calcje.
%    
%    grad - numToCheck x 1 double vector. The computed gradients at the
%    indices provided by "ind".
% 
% where N is the number of examples
% M is the side length of one example (assuming square images)
% K is the number of output classes

[numgrad,~] = check_finit_dif(cnet,num,Ip,d,order,numToCheck,ind);
grad = je(ind);
diff = arrayfun(@(ix) compareGrads(numgrad(ix), grad(ix)), 1:length(numgrad));

end

function [diff] = compareGrads(ng, g)
    
    % To handle numerical precision issues, first check if the numerical 
    % approximation is zero (in which case MATLAB failed to compute a 
    % difference in MSE for the two perturbed weights) AND if the computed
    % gradient is sufficiently close to zero as well -> in this case, the
    % two gradients are sufficiently equal. Otherwise, use the standard
    % norm comparison.

    %Tolerance for determining if two floating-point numbers are equal
%     tol = 10^-8;
%     
%     if abs(ng-0) <= tol && abs(g-0) <= tol
%         diff = tol;
%     else
%         diff = norm(ng-g)/norm(ng+g);
%     end
    
    diff = norm(ng-g)/norm(ng+g);

end