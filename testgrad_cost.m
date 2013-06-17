function [cost] = testgrad_cost(cnet,d)
%
% Given the simulated network cnet, and desired targets d, compute the mean
% squared error cost function.
%

out = cnet.FLayer.X;
cost = sum((out - d) .^ 2) / cnet.FLayer.numOutputs;

end