function [out, indices] = order(in, ratio, operation)
% Apply an ordering (sorting) operation to a ratio x ratio region in the
% input, in.
% 
% INPUTS
% in - the input data
% ratio - integer, representing the side length of the region to process
% operation - string, representing the type of ordering operation.
% { 'descend', 'ascend' }. By default, sorting is done column-wise, e.g.:
%
% in
% [ 3   6   1;
%   2   2   9;
%   8   4   5]
%
% out
% [ 6   2   9   2;
%   3   2   6   1;
%   8   2   9   4;
%   4   2   5   2]
%
% indices
% [ 4   2   8   5;
%   1   5   4   7;
%   3   2   8   6;
%   6   5   9   5]
%
%
% OUTPUTS
% out - the data with ordered ratio x ratio regions. The regions overlap by
% stride=1, therefore the output is larger.
% indices - the index of each element in the original matrix
%

if nargin < 3
    operation = 'descend'; % default: sort in descending order
end

% The size of the output
out = nan((size(in,1)-ratio+1)*2,(size(in,2)-ratio+1)*2);
indices = nan((size(in,1)-ratio+1)*2,(size(in,2)-ratio+1)*2);

% Assuming a stride of 1
for k=1:size(in,1)-ratio+1
    for m=1:size(in,2)-ratio+1
        % Sort the next region
        nextRegion = in(k:k+ratio-1,m:m+ratio-1);
        
        % Compute the linear indices of the elements in nextRegion in the
        % original input matrix, in
        combos = combinations(k:k+ratio-1,m:m+ratio-1,2);
        linind = sort( sub2ind(size(in), combos(:,1), combos(:,2) )); % col-wise
        
        % ix are the indices within nextRegion => convert to indices within
        % the input matrix by subscripting "linind"
        [sortedRegion, ix] = sort(nextRegion(:), operation);
        indices((k-1)*ratio+1:k*ratio,(m-1)*ratio+1:m*ratio) = reshape(linind(ix),ratio,ratio);
        out((k-1)*ratio+1:k*ratio,(m-1)*ratio+1:m*ratio) = reshape(sortedRegion,ratio,ratio);
    end
end        

end