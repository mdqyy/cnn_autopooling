 function out = back_subsample(e, ratio, operation, indices, weights) 
 %back_subsample backpropagate error through subsample layers 
%
%  Syntax
%  
%    out = back_subsample(e, ratio)
%    
%  Description
%   Input:
%    e - error map (dEdSS - error wrt subsampled inputs to SLayer)
%    ratio - expansion ratio
%    operation - { 'average', 'max', 'stochastic', 'auto' }
%    indices - index of every element from SS in the original matrix X 
%    before subsampling; required if the operation is max or stochastic
%    weights - ratio x ratio matrix. Required if the operation is auto
%
%   Output:
%    out - backpropagated error map with size ratio*size(e) (dEdX for the
%    previous CLayer, given that SS = subsample(X, ratio))
%
%(c) Sirotenko Mikhail, 2009

if nargin < 3
    operation = 'average'; % default operation
end

if ratio == 1
    out = e;
else
    if strcmp(operation, 'average') == 1
        
        out = 0;
        for k=1:ratio
            for l=1:ratio
                out(1+(k-1):ratio:size(e,1)*ratio,1+(l-1):ratio:size(e,2)*ratio) = e;
            end
        end
        out = out ./ (ratio*ratio);
        
    elseif strcmp(operation, 'max') == 1 || strcmp(operation, 'stochastic') == 1
        % SS = max(x1, ..., xN) where N = ratio*ratio
        % Only the max element (the one at the index specified in
        % "indices") will get propagated, the rest are zero
        out = zeros(ratio*size(e));
        out(indices) = e;
        
    elseif strcmp(operation, 'auto') == 1
        % SS = X * weights + bias
        % dEdX = dEdS * weights
        weights = reshape(weights, ratio, ratio);
        out = zeros(ratio*size(e));
        for k=1:ratio
            for l=1:ratio
                out(1+(k-1):ratio:size(e,1)*ratio,1+(l-1):ratio:size(e,2)*ratio) = e * weights(k,l);
            end
        end
    end
end

   
end