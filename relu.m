function out = relu(in1,in2,in3)
%RELU Rectified linear unit transfer function. 
% 
% Syntax
%
%   A = relu(N)
%   dA_dN = relu('dn',N,A)
%
% Description
% 
%   RELU is a neural transfer function.  Transfer functions
%   calculate a layer's output from its net input.
% 
%   RELU('dn',N,A) returns derivative of A w-respect to N.
%
%(c) Maria Yancheva, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Apply Transfer Function

if strcmp(in1,'dn')
    out = derivative(in2,in3);
else
    out = apply_transfer(in1);
end

function a = apply_transfer(n)
if n > 0
    a = n;
else
    a = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Derivative of Y w/respect to X
function da_dn = derivative(n,a)
if n > 0
    da_dn = 1;
else
    da_dn = 0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
