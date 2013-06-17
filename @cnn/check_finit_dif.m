function [num_grad, ind_grad] = check_finit_dif(cnet,num,Ip,d,order,numToCheck,ind) 
%check_finit_dif calculate gradient or Hessian using finite differences
%
%  Syntax
%  
%    difference = check_finit_dif(cnet,num,Ip,d,order) 
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    num - number of parameters (weights and biases) in single-column weight
%    vector (which have to be learned)
%    Ip - K x K double, 1 input example
%    d - desired output, 1 x N
%    order - 1 means gradient, 2 - Hessian
%    numToCheck - 1 x 1 int, the number of gradients to check (since
%    checking all gradients in the network takes too long for deep nets)
%    ind - 1 x N int, the indices of the gradients to be checked
%    numerically (e.g., ind = [1] means check dEdW (wrt w1) for last layer)
%    
%   Output:
%    num_grad - numToCheck x 1 double vector. Gradients for all parameters, 
%    calculated using finite differences. If calcje works correctly, 
%    the numerical gradients should be almost the same as the gradients 
%    produced by calcje.
%    ind_grad - numToCheck x 1 int vector. The indices of the gradients
%    that were computed numerically (if numToCheck < num, the gradients 
%    are randomly sampled to save time)
%   
%   Description:
%    This function is mostly used for debugging, because calculating
%    gradients such was is computationally expensive
%
% where N is the number of output classes (e.g., N=10 for digit recognition)
% K is the size of one example (e.g., K=32 for MNIST images)
% 

%Epsilon determines the accuracy of this method
epsilon = 10^-4;        
if ~isnan(ind)
    numToCheck = length(ind);
end

switch(order)
    case 1 % Gradient
        % Initialize the numerical gradient
        num_grad = zeros(numToCheck,1);
        perturb = zeros(num,1);
        
        % Too slow to compute ALL gradients numerically, so pick randomly
        % which ones to check
        %ind_grad = round( 1 + rand(numToCheck,1)*(num-1) );
        ind_grad = ind;
        
        for p=1:numToCheck
            
            p_ind = ind_grad(p);
            
            % Set perturbation vector (only perturb 1 weight at a time)
            perturb(p_ind) = epsilon;
            
            % Simulate the net with different error values
            cnet_minus_e = adapt_dw(cnet,perturb);
            cnet_plus_e = adapt_dw(cnet,-perturb);
            
            out1 = sim(cnet_plus_e,Ip);
            out2 = sim(cnet_minus_e,Ip);
            
            e1 = out1-d;
            e2 = out2-d;
            
            % Compute numerical gradient
            num_grad(p) = (mse(e1)-mse(e2)) / (2*epsilon);
            perturb(p_ind) = 0;
        end
        
    case 2 % Hessian. Not working properly yet
%         dW = sparse(zeros(20691,1));
%         dW(num) = 2*epsilon;
%         cnet_minus_e = adapt_dw(cnet,dW);
%         cnet_plus_e = adapt_dw(cnet,-dW);
%         e1 = sim(cnet_plus_e,Ip)-d;
%         e2 = sim(cnet_minus_e,Ip)-d;
%         e3 = sim(cnet,Ip)-d;
%         d2Ed2Wi = (mse(e1)+mse(e2)-2*mse(e3))/(4*epsilon^2);
%         difference = d2Ed2Wi;

%RECURSIVE
%         dW2 = sparse(zeros(20691,1));
%         dW2(num) = epsilon;
%         cnet_minus_e2 = adapt_dw(cnet,dW2);
%         cnet_plus_e2 = adapt_dw(cnet,-dW2);
%         je1 = check_finit_dif(cnet_plus_e2,num,Ip,d,1);
%         je2 = check_finit_dif(cnet_minus_e2,num,Ip,d,1);
%         d2Ed2Wi2 = (je1-je2)/(2*epsilon);
%         difference = d2Ed2Wi;

        
    otherwise
        error('Order should be 1 or 2');
end