function out = mse(in1, in2, in3)
%MSE Mean squared performance function
% 
% Syntax
%
%   perf = mse(predictions,targets)
%   predictions: N x K vector of doubles
%   targets: N x K vector of one-hot encoded target values
%   Returns the cost, 1x1 double
%
%   dPerf_dy = mse('dy',predictions,targets)
%   predictions: N x K vector of doubles
%   targets: N x K vector of one-hot encoded target values
%   Returns the derivative of the cost with respect to the output
%   predictions, N x K
%
%   where K is the number of output classes
%   N is the number of examples
%
% Description
% 
%   mse is a network performance function. It measures the network's performance according to the mean of squared errors.
%   This is very simple replacement of the NN toolkit version.
% 

    if strcmp(in1,'dy')
        out = derivative(in2,in3);
    else
        out = apply_perf(in1,in2);
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Apply 
function a = apply_perf(predictions,targets)
    error = predictions - targets;
    a = sum(sum(error.^2))/size(error,1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Derivative 
function dout_dinp = derivative(predictions,targets)
    error = predictions - targets;
    dout_dinp = 2*error/size(error,1);
end
