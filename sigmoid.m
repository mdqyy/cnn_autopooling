function out = sigmoid(in1,in2,in3)
%SIGMOID - sigmoid transfer function. 
% 
% INPUTS
% in1 - inputs to apply function (1 x N) to OR 'dn'
% If 'dn' specified, the function returns a derivative of in3 wrt in2
%
% Syntax
%
%   A = sigmoid(N)
%   dA_dN = sigmoid('dn',N,A)
%
% Description
% 
%   SIGMOID is a neural transfer function.  Transfer functions
%   calculate a layer's output from its net input.
% 
%   SIGMOID('dn',N,A) returns derivative of A w-respect to N.
%
%(c) Maria Yancheva, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Apply Transfer Function

if strcmp(in1,'dn')
    out = derivative(in2,in3);
else
    out = apply_transfer(in1);
end

function a = apply_transfer(n)
    % a = 1 / ( 1 + e^(-x) )
    a = 1.0 ./ (ones(size(n)) + exp(-n));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Derivative of Y w/respect to X
function da_dn = derivative(n,a)
    % da/dn = a .* (1 - a) iff a = sigmoid(n)
    da_dn = a .* (1 - a);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
