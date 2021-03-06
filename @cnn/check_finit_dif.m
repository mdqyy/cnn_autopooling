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
%    Ip - M x M double, 1 input example
%    d - desired output, 1 x N
%    order - 1 means gradient, 2 - Hessian
%    numToCheck - 1 x 1 int, the number of gradients to check (since
%    checking all gradients in the network takes too long for deep nets)
%    ind - 1 x numToCheck int, the indices of the gradients to be checked
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
%    gradients is computationally expensive
%
% where K is the number of output classes (e.g., N=10 for digit recognition)
% M is the size of one example (e.g., M=32 for MNIST images)
% N is the number of training examples

% Delta determines the accuracy of this method (the smaller the
% perturbation, the closer this approximation would be to the actual 
% gradient)
delta = 10^-4;

% Convert labels to one-hot encoding, N x K matrix
numPats = length(Ip);
targets = repmat(1:cnet.numOutputs, numPats, 1) == repmat(d+1,1,cnet.numOutputs);

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
            perturb(p_ind) = delta;
            
            % Simulate the net with different error values
            cnet_minus_e = adapt_dw(cnet,perturb);
            cnet_plus_e = adapt_dw(cnet,-perturb);
            
            out1 = sim(cnet_plus_e,Ip); % N x K output
            out2 = sim(cnet_minus_e,Ip); % N x K output
            
            % Use the error function of the CNN to evaluate performance
            e1 = feval(cnet.Perf, out1, targets);
            e2 = feval(cnet.Perf, out2, targets);
            
            % Compute numerical gradient
            num_grad(p) = (e1-e2) / (2*delta);
            perturb(p_ind) = 0;
        end
        
    case 2 % Hessian. Not working properly yet
%         dW = sparse(zeros(20691,1));
%         dW(num) = 2*delta;
%         cnet_minus_e = adapt_dw(cnet,dW);
%         cnet_plus_e = adapt_dw(cnet,-dW);
%         e1 = sim(cnet_plus_e,Ip)-d;
%         e2 = sim(cnet_minus_e,Ip)-d;
%         e3 = sim(cnet,Ip)-d;
%         d2Ed2Wi = (mse(e1)+mse(e2)-2*mse(e3))/(4*delta^2);
%         difference = d2Ed2Wi;

%RECURSIVE
%         dW2 = sparse(zeros(20691,1));
%         dW2(num) = delta;
%         cnet_minus_e2 = adapt_dw(cnet,dW2);
%         cnet_plus_e2 = adapt_dw(cnet,-dW2);
%         je1 = check_finit_dif(cnet_plus_e2,num,Ip,d,1);
%         je2 = check_finit_dif(cnet_minus_e2,num,Ip,d,1);
%         d2Ed2Wi2 = (je1-je2)/(2*delta);
%         difference = d2Ed2Wi;

        
    otherwise
        error('Order should be 1 or 2');
end