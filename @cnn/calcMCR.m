function [mcr, mse, cnet] = calcMCR(cnet,I_testp,labels, idxs)
%calcMCR Calculate missclassification rate
%
%  Syntax
%  
%    mcr = calcMCR(cnet,It,labels, idxs)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    I_testp - cell array, containing preprocessed images of handwriten digits
%    labels - cell array of labels, corresponding to images
%    idxs - vector of indices of elements to use for testing
%   Output:
%    mcr - missclassification rate
%    mse - mean squared error on the provided dataset
%    cnet - simulated CNN with stored outputs
%
%(c) Sirotenko Mikhail, 2009
correct=0;
error=0.0;
for i=idxs
    [out, cnet] = sim(cnet,I_testp{i});  
    
    % Display the (predicted, actual) labels
    %fprintf('Output probabilities:\n');
    %disp(out(:) ./ sum(out(:)));
    %fprintf('\nPredicted digit: %d, Actual digit: %d\n', find(out==max(out))-1, labels(i));
    
    % out is the output of probabilities for each class => compare the
    % index of the maximum probability class with the target label
    if(find(out==max(out))==(labels(i)+1))
        correct=correct+1;
    end
    
    %Compute MSE by adding up squared error for each data point
    targets = zeros(1,10);
    targets(labels(i)+1) = 1;
    out = out ./ sum(out);
    e = out - targets;
    error = error + e*e';
end
mcr = 1-correct/length(idxs);
mse = error/length(idxs);