function out = negloglik(in1, in2, in3)
% negloglik - computes the negative log likelihood cost for the
% provided data, or (if the first argument is 'dy') compute the derivative
% fo the NLL cost function wrt the input.
%
% INPUTS
% perf = negloglik(predictions,targets)
% dPerfdY = negloglik('dy', predictions, targets)
%
% predictions = N x K
% targets = N x K
%
% where N is the number of training examples
% and K is the number of output classes
%
% OUTPUTS
% out - either NLL (1x1) or derivative of NLL (N x K)
%

    if strcmp(in1,'dy')
        out = derivative(in2,in3);
    else
        out = apply_perf(in1,in2);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Apply 
    function a = apply_perf(h, targets)
        a = 1/size(h,1) * sum( sum( -targets.*log(h) -(1-targets).*log(1-h) ) );
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Derivative
    function dout_dinp = derivative(h, targets)
        dout_dinp = 1/size(h,1) * ( -targets./h + (1-targets)./(1-h) );
    end


end