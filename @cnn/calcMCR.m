function [mcr, mse, cnet, pred_labels] = calcMCR(cnet,I_testp,labels, idxs)
%calcMCR Calculate missclassification rate
%
%  Syntax
%  
%    mcr = calcMCR(cnet,It,labels, idxs)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network class object
%    I_testp - cell array, containing preprocessed images of handwriten
%    digits, 1 x N
%    labels - cell array of labels, N x 1, corresponding to images
%    idxs - vector of indices of elements to use for testing
%   Output:
%    mcr - missclassification rate
%    mse - mean squared error on the provided dataset
%    cnet - simulated CNN with stored outputs
%    pred_labels - N x K matrix of output probabilities
%
% where N is the number of input images
% and K is the number of output classes
%

correct=0;
error=nan(length(labels),1);
pred_labels = nan(length(labels), cnet.numOutputs);
target_labels = nan(length(labels), cnet.numOutputs);

for i=idxs
    [out, cnet] = sim(cnet,I_testp(i));  
    
    % out is the output of probabilities for each class => compare the
    % index of the maximum probability class with the target label
    if(find(out==max(out))==(labels(i)+1))
        correct=correct+1;
    end
    
    %Compute MSE by adding up squared error for each data point
    targets = zeros(1,cnet.numOutputs);
    targets(labels(i)+1) = 1;
    
    pred_labels(i, :) = out;
    target_labels(i, :) = targets;
    
    %out = out ./ sum(out);
    e = out - targets;
    error(i) = sum(e.^2);
    
end
mcr = 1-correct/length(idxs);
mse = sum(error)/length(labels);

