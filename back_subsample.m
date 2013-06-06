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

    switch ratio
        case 4
            if strcmp(operation, 'average') == 1
                out = 0;
                for k=1:4
                    for l=1:4
                        out(1+(k-1):4:size(e,1)*4,1+(l-1):4:size(e,2)*4) = e;
                    end
                end
                out = out.*0.0625;
            elseif strcmp(operation, 'max') == 1 || strcmp(operation, 'stochastic') == 1
                % SS = max(x1, ..., xN) where N = ratio*ratio
                % Only the max element (the one at the index specified in
                % "indices") will get propagated, the rest are zero
                out = zeros(ratio*size(e));
                out(indices) = e;
                
            elseif strcmp(operation, 'auto') == 1
                % SS = X * weights + bias
                % dEdX = dEdS * weights
                out = zeros(ratio*size(e));
                for k=1:4
                    for l=1:4
                        out(1+(k-1):4:size(e,1)*4,1+(l-1):4:size(e,2)*4) = e * weights(k,l);
                    end
                end
            end

        case 2
            if strcmp(operation, 'average') == 1
                out(1:2:size(e,1)*2,1:2:size(e,2)*2)=e;
                out(1:2:size(e,1)*2,2:2:size(e,2)*2)=e;
                out(2:2:size(e,1)*2,1:2:size(e,2)*2)=e;
                out(2:2:size(e,1)*2,2:2:size(e,2)*2)=e;
                out=out.*0.25;
            elseif strcmp(operation, 'max') == 1 || strcmp(operation, 'stochastic') == 1
                % SS = max(x1, ..., xN) where N = ratio*ratio
                % Only the max element (the one at the index specified in
                % "indices") will get propagated, the rest are zero
                out = zeros(ratio*size(e));
                out(indices) = e;
                
            elseif strcmp(operation, 'auto') == 1
                % SS = X * weights + bias
                % dEdX = dEdS * weights
                % Assume weights are used column-wise, e.g.:
                % [1 4
                %  2 3] is equivalent to using [1 2 4 3]
                %
                out(1:2:size(e,1)*2,1:2:size(e,2)*2)=e * weights(1);
                out(2:2:size(e,1)*2,1:2:size(e,2)*2)=e * weights(2);
                out(1:2:size(e,1)*2,2:2:size(e,2)*2)=e * weights(3);
                out(2:2:size(e,1)*2,2:2:size(e,2)*2)=e * weights(4);
                
            end
        case 1
            out = e;
        otherwise
            disp('Unsupported ratio');
    end

   
end