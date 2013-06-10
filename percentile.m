function [out, indices] = percentile(x, p)
%
% For the purposes of taking percentiles in a CNN, we need a value that
% exists in "x" (in order to be able to do backprop). The built-in matlab
% implementation of "prctile" returns a value that may not exist in "x".
% This custom implementation computes the percentile as follows:
% 1. ind (rounded to nearest integer) = p/100 x N + 1/2
% 2. out = x(ind) 
%
% where N is the number of values in "x"
% M is the number of percentiles we need
%
% INPUTS
% x - a vector of doubles, N x 1
% p - a vector of doubles, 1 x M
%
% OUTPUTS
% out - a vector of values (which exist in x), M x 1
% indices - the indices in the original input "x" of the values that are
% returned as percentiles in "out", M x 1
%

N = length(x);
[x, ix] = sort(x);
ind = round( p/100 * N + 0.5 );
out = x(ind);
indices = ix(ind);

end