function mcr = calcMCR(cnet,I_testp,labels, idxs)
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
%
%(c) Sirotenko Mikhail, 2009
correct=0;
for i=idxs
    [out, cnet] = sim(cnet,I_testp{i});  
    % out is the output of probabilities for each class => compare the
    % index of the maximum probability class with the target label
    if(find(out==max(out))==(labels(i)+1))
        correct=correct+1;
    end
end
mcr = 1-correct/length(idxs);